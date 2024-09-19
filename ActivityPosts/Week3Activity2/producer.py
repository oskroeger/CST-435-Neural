# producer.py
from websocket_server import WebsocketServer
import json
import time
import random
from datetime import datetime

def new_client(client, server):
    print(f"New client connected and was given id {client['id']}")
    send_random_data(server)

def is_fraudulent(transaction):
    """Simple rules to determine if a transaction is fraudulent."""
    if transaction['userID'] < 10 and transaction['amount'] > 300:  # New user with high amount
        return 1
    elif transaction['amount'] > 400:  # High transaction amount in general
        return 1
    else:
        return 0

def send_random_data(server):
    transaction_id = 1
    while True:
        # Simulate transaction data
        data = {
            "transactionID": transaction_id,
            "userID": random.randint(1, 100),
            "amount": round(random.uniform(5.0, 500.0), 2),
            "timestamp": datetime.now().isoformat(),
            "itemID": f"E{random.randint(1000, 9999)}"
        }
        # Apply simple fraud rules
        data["fraud_label"] = is_fraudulent(data)
        server.send_message_to_all(json.dumps(data))
        print(f"Sent: {data}")
        transaction_id += 1
        time.sleep(2)  # Send data every 2 seconds

if __name__ == "__main__":
    server = WebsocketServer(port=9999, host='127.0.0.1')
    server.set_fn_new_client(new_client)
    server.run_forever()
