from flask import Flask, request, jsonify
from pymongo import MongoClient
import pymongo.errors
import os
from flask_cors import CORS
from Fraud_detection import FraudDetector  # Import the fraud detection module

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}}, supports_credentials=True, 
     allow_headers=["Content-Type", "Authorization"], methods=["GET", "POST", "OPTIONS"])

# MongoDB Connection
mongo_uri = "mongodb+srv://Khelan05:KrxRwjRwkhgYUdwh@cluster0.c6y9phd.mongodb.net/fd1?retryWrites=true&w=majority"
try:
    client = MongoClient(mongo_uri)
    db = client['fd1']
    transactions_collection = db['transactions']
    predictions_collection = db['predictions']
    print("Connected to MongoDB successfully!")
except pymongo.errors.ConnectionError as e:
    print(f"Failed to connect to MongoDB: {e}")
    exit(1)

# Initialize Fraud Detector
model_path = "model.pth"
csv_path = "modified_dataset.csv"
detector = FraudDetector(model_path, csv_path)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Fraud Detection API. Use POST /predict to submit transactions."}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        transaction = request.get_json()
        required_fields = ['Sender_account', 'Receiver_account', 'Amount', 
                          'Sender_bank_location', 'Receiver_bank_location']
        if not all(field in transaction for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Process transaction using FraudDetector
        result = detector.process_transaction(transaction)
        
        # Store in MongoDB
        transaction_doc = {
            "Sender_account": transaction['Sender_account'],
            "Receiver_account": transaction['Receiver_account'],
            "Amount": transaction['Amount'],
            "Sender_bank_location": transaction['Sender_bank_location'],
            "Receiver_bank_location": transaction['Receiver_bank_location'],
            "Timestamp": transaction.get("Timestamp", None)
        }
        transaction_id = transactions_collection.insert_one(transaction_doc).inserted_id
        
        prediction_doc = {
            "Sender_account": transaction['Sender_account'],
            "Is_laundering": result["is_laundering"],
            "Laundering_type": result["type"],
            "Transaction_id": str(transaction_id)
        }
        predictions_collection.insert_one(prediction_doc)
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Handle Preflight Requests
@app.route('/predict', methods=['OPTIONS'])
def handle_preflight():
    response = jsonify({'message': 'CORS preflight successful'})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
    return response, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
