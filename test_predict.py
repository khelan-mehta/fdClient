import requests

transaction = {
    "Sender_account": 123,
    "Receiver_account": 456,
    "Amount": 5000,
    "Sender_bank_location": "Nigeria",
    "Receiver_bank_location": "USA",
    "Timestamp": "2023-10-25T10:00:00"
}

response = requests.post("http://10.30.65.21:5000/predict", json=transaction)
print(response.status_code, response.json())