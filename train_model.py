import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Define GNN Model
class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Load dataset (adjust the file path to your CSV)
df = pd.read_csv("modified_dataset.csv")  # Replace with your actual file path

# Encode accounts
account_encoder = LabelEncoder()
all_accounts = pd.concat([df['Sender_account'], df['Receiver_account']]).unique()
account_encoder.fit(all_accounts)
df['Sender_account'] = account_encoder.transform(df['Sender_account'])
df['Receiver_account'] = account_encoder.transform(df['Receiver_account'])

# Encode locations
location_encoder = LabelEncoder()
all_locations = pd.concat([df['Sender_bank_location'], df['Receiver_bank_location']]).unique()
location_encoder.fit(all_locations)
df['Sender_bank_location'] = location_encoder.transform(df['Sender_bank_location'])
df['Receiver_bank_location'] = location_encoder.transform(df['Receiver_bank_location'])

# Graph Construction for Cycle Detection
def build_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Sender_account'], row['Receiver_account'], amount=row['Amount'])
    return G

def detect_cycles(graph):
    cycles = list(nx.simple_cycles(graph))
    return set([node for cycle in cycles for node in cycle])

G = build_graph(df)
cyclic_accounts = detect_cycles(G)
df['Cycle_Flag'] = df['Sender_account'].apply(lambda x: 1 if x in cyclic_accounts else 0)

# Smurfing Detection with Isolation Forest
iso_forest = IsolationForest(contamination=0.1)
df['Smurfing_Score'] = iso_forest.fit_predict(df[['Amount']])
df['Smurfing_Flag'] = df['Smurfing_Score'].apply(lambda x: 1 if x == -1 else 0)

# Location-Based Fraud Detection
suspicious_locations = set(location_encoder.transform(['Nigeria', 'UAE', 'Pakistan']))
df['Location_Flag'] = df.apply(
    lambda x: 1 if x['Sender_bank_location'] in suspicious_locations or x['Receiver_bank_location'] in suspicious_locations else 0, 
    axis=1
)

# Prepare node features
num_nodes = len(account_encoder.classes_)
print("Number of unique accounts (nodes):", num_nodes)

node_features = np.zeros((num_nodes, 5))  # 5 features: mean Amount, transaction count, Cycle_Flag, Smurfing_Flag, Location_Flag
for account in range(num_nodes):
    sent = df[df['Sender_account'] == account]
    received = df[df['Receiver_account'] == account]
    total_amount = pd.concat([sent['Amount'], received['Amount']])
    node_features[account, 0] = total_amount.mean() if not total_amount.empty else 0.0  # Mean Amount
    node_features[account, 1] = len(sent) + len(received)  # Transaction count
    node_features[account, 2] = 1 if account in cyclic_accounts else 0  # Cycle_Flag
    node_features[account, 3] = sent['Smurfing_Flag'].mean() if not sent.empty else 0  # Smurfing_Flag (mean)
    node_features[account, 4] = sent['Location_Flag'].mean() if not sent.empty else 0  # Location_Flag (mean)

x = torch.tensor(node_features, dtype=torch.float)

# Create edge_index
edge_index = torch.tensor(
    [[df['Sender_account'][i], df['Receiver_account'][i]] for i in range(len(df))], 
    dtype=torch.long
).t().contiguous()

# Use Is_laundering as the target
y = torch.zeros(num_nodes, dtype=torch.long)
for i, row in df.iterrows():
    if row['Is_laundering'] == 1:
        y[row['Sender_account']] = 1  # Mark sender as laundering

# Train/test split
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_size = int(0.8 * num_nodes)
train_indices = np.random.choice(num_nodes, train_size, replace=False)
test_indices = np.setdiff1d(np.arange(num_nodes), train_indices)
train_mask[train_indices] = True
test_mask[test_indices] = True

# Data object
data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

# Class weights to handle imbalance
class_counts = torch.bincount(y)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()

# Train Model
model = GNN(num_features=5, hidden_dim=16, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Training loop
for epoch in range(200):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "model.pth")
print("Model trained and saved as 'model.pth'")