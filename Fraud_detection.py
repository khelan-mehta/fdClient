import torch
import networkx as nx
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import os
import pandas as pd
from collections import Counter

class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class FraudDetector:
    def __init__(self, model_path, csv_path):
        self.G = nx.DiGraph()
        self.account_features = {}
        self.node_to_idx = {}
        
        self.account_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            self.df['Sender_account'] = self.df['Sender_account'].astype(str)
            self.df['Receiver_account'] = self.df['Receiver_account'].astype(str)
            
            all_accounts = pd.concat([self.df['Sender_account'], self.df['Receiver_account']]).unique()
            self.account_encoder.fit(all_accounts)
            self.df['Sender_account_encoded'] = self.account_encoder.transform(self.df['Sender_account'])
            self.df['Receiver_account_encoded'] = self.account_encoder.transform(self.df['Receiver_account'])
            
            all_locations = pd.concat([self.df['Sender_bank_location'], self.df['Receiver_bank_location']]).unique()
            self.location_encoder.fit(all_locations)
            self.df['Sender_bank_location'] = self.location_encoder.transform(self.df['Sender_bank_location'])
            self.df['Receiver_bank_location'] = self.location_encoder.transform(self.df['Receiver_bank_location'])
            
            self.df['Amount'] = self.scaler.fit_transform(self.df[['Amount']])
            
            for _, row in self.df.iterrows():
                sender = row['Sender_account']
                receiver = row['Receiver_account']
                self.G.add_edge(sender, receiver, amount=row['Amount'])
                self.node_to_idx[sender] = sender
                self.node_to_idx[receiver] = receiver
            
            self.cyclic_accounts = set([node for cycle in nx.simple_cycles(self.G) for node in cycle])
            print(f"Preloaded graph with {self.G.number_of_edges()} edges from CSV.")
            print(f"Location Encoder Classes: {self.location_encoder.classes_}")
            
            self.iso_forest = IsolationForest(contamination=0.1)
            self.df['Smurfing_Score'] = self.iso_forest.fit_predict(self.df[['Amount']])
            self.df['Smurfing_Flag'] = self.df['Smurfing_Score'].apply(lambda x: 1 if x == -1 else 0)
            
            self.account_features = {account: [0.0, 0, 0, 0, 0] for account in all_accounts}
            for account in all_accounts:
                sent = self.df[self.df['Sender_account'] == account]
                received = self.df[self.df['Receiver_account'] == account]
                total_amount = pd.concat([sent['Amount'], received['Amount']])
                self.account_features[account][0] = total_amount.mean() if not total_amount.empty else 0.0
                self.account_features[account][1] = len(sent) + len(received)
                self.account_features[account][2] = 1 if account in self.cyclic_accounts else 0
                self.account_features[account][3] = sent['Smurfing_Flag'].mean() if not sent.empty else 0
        else:
            raise FileNotFoundError("CSV file not found.")
        
        self.num_features = 5
        self.model = GNN(self.num_features, hidden_dim=16, num_classes=2)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Model file {model_path} not found.")
        self.model.eval()

    def process_transaction(self, transaction):
        sender = str(transaction['Sender_account'])
        receiver = str(transaction['Receiver_account'])
        amount = transaction['Amount']
        sender_location = transaction['Sender_bank_location']
        receiver_location = transaction['Receiver_bank_location']

        def safe_transform(encoder, value):
            if value not in encoder.classes_:
                new_classes = np.append(encoder.classes_, value)
                encoder.classes_ = new_classes
            return encoder.transform([value])[0]

        sender_idx = self.account_encoder.transform([sender])[0]
        receiver_idx = self.account_encoder.transform([receiver])[0]
        amount_scaled = self.scaler.transform([[amount]])[0][0]
        sender_loc_idx = safe_transform(self.location_encoder, sender_location)
        receiver_loc_idx = safe_transform(self.location_encoder, receiver_location)

        if sender not in self.node_to_idx:
            self.node_to_idx[sender] = sender
        if receiver not in self.node_to_idx:
            self.node_to_idx[receiver] = receiver
        
        self.G.add_edge(sender, receiver, amount=amount_scaled)

        # Backtracking functions
        def trace_location_pattern(account, n=3):
            sent = self.df[self.df['Sender_account'] == account].tail(n)
            if len(sent) < n:
                return None, False
            sender_locs = list(sent['Sender_bank_location'])
            receiver_locs = list(sent['Receiver_bank_location'])
            most_common_receiver = Counter(receiver_locs).most_common(1)[0][0]
            sender_consistent = all(loc == sender_locs[0] for loc in sender_locs)
            print(f"Location Pattern - Sender Locs: {sender_locs}, Receiver Locs: {receiver_locs}")
            return most_common_receiver, sender_consistent

        def trace_smurfing_pattern(account, n=3, threshold=0.5):
            sent = self.df[self.df['Sender_account'] == account].tail(n)
            if len(sent) < n:
                return False
            amounts = list(sent['Amount'])
            print(f"Smurfing Pattern - Last {n} Amounts: {amounts}")
            return all(a < threshold for a in amounts)

        # Cyclic detection with backtracking
        cycle_flag = 0
        cyclic_accounts = set([node for cycle in nx.simple_cycles(self.G) for node in cycle])
        if sender in cyclic_accounts:
            predecessors = list(self.G.predecessors(sender))
            if receiver in predecessors or any(self.G.has_edge(pred, receiver) for pred in predecessors):
                cycle_flag = 1
                print(f"Cyclic Fraud Detected: Transaction forms part of a cycle, Predecessors: {predecessors}")

        # Smurfing detection with backtracking
        smurfing_score = self.iso_forest.predict([[amount_scaled]])[0]
        smurfing_flag = 0
        sent = self.df[self.df['Sender_account'] == sender]
        if trace_smurfing_pattern(sender) and amount_scaled < 0.5:
            smurfing_flag = 1
            print(f"Smurfing Fraud Detected: Last transactions all small, Current: {amount_scaled}")
        elif smurfing_score == -1 and amount_scaled < 0.5:
            smurfing_flag = 1
            print(f"Smurfing Fraud Detected: Outlier small amount")

        # Location-based fraud with backtracking
        most_frequent_receiver_loc, sender_consistent = trace_location_pattern(sender)
        tx_count = len(sent)
        location_flag = 0
        if tx_count >= 3 and most_frequent_receiver_loc is not None and sender_consistent:
            if receiver_loc_idx != most_frequent_receiver_loc:
                location_flag = 1
                print(f"Location Fraud Detected: Pattern to {most_frequent_receiver_loc} ({self.location_encoder.inverse_transform([most_frequent_receiver_loc])[0]}), now {receiver_loc_idx} ({receiver_location})")

        # Update transaction features
        if sender not in self.account_features:
            self.account_features[sender] = [0.0, 0, 0, 0, 0]
        
        sent = self.df[self.df['Sender_account'] == sender].copy()
        received = self.df[self.df['Receiver_account'] == sender].copy()
        new_tx = pd.DataFrame({
            'Sender_account': [sender], 'Receiver_account': [receiver], 
            'Amount': [amount_scaled], 'Smurfing_Flag': [smurfing_flag], 'Location_Flag': [location_flag]
        })
        sent = pd.concat([sent, new_tx])
        total_amount = pd.concat([sent['Amount'], received['Amount']])
        self.account_features[sender][0] = total_amount.mean() if not total_amount.empty else 0.0
        self.account_features[sender][1] = len(sent) + len(received)
        self.account_features[sender][2] = cycle_flag
        self.account_features[sender][3] = sent['Smurfing_Flag'].mean() if not sent.empty else 0
        self.account_features[sender][4] = location_flag

        # Prepare GNN input
        all_accounts = list(self.account_features.keys())
        x = torch.tensor([self.account_features[account] for account in all_accounts], dtype=torch.float)
        edge_index = torch.tensor(
            [[self.account_encoder.transform([e[0]])[0], self.account_encoder.transform([e[1]])[0]] for e in self.G.edges()],
            dtype=torch.long
        ).t().contiguous()

        with torch.no_grad():
            out = self.model(x, edge_index)
            sender_pos = all_accounts.index(sender)
            gnn_is_laundering = out[sender_pos].argmax().item()

        # Populate laundering types
        laundering_type = []
        if cycle_flag:
            laundering_type.append("Cyclic")
        if smurfing_flag:
            laundering_type.append("Smurfing")
        if location_flag:
            laundering_type.append("Location_Based_Fraud")

        is_laundering = bool(len(laundering_type) > 0)

        # Debug prints
        print(f"Sender: {sender}, Receiver: {receiver}, Amount: {amount}")
        print(f"Cycle Flag: {cycle_flag}")
        print(f"Smurfing Score: {smurfing_score}, Smurfing Flag: {smurfing_flag}, Amount Scaled: {amount_scaled}")
        print(f"Location Flag: {location_flag}, Sender Loc: {sender_location}, Receiver Loc: {receiver_location}")
        print(f"Most Frequent Receiver Loc: {most_frequent_receiver_loc}, Tx Count: {tx_count}")
        print(f"Laundering Type: {laundering_type}")
        print(f"Is Laundering (GNN): {bool(gnn_is_laundering)}")
        print(f"Final Is Laundering: {is_laundering}")

        return {"is_laundering": is_laundering, "type": laundering_type if laundering_type else []}