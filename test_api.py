import os
import django
import json

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.test import Client

client = Client(HTTP_HOST='localhost')

print("--- Testing API Endpoint ---")

# Test 1: Empty string
response = client.post('/api/v1/classify/', content_type="application/json", data=json.dumps({"text": ""}))
if response.status_code == 400:
    print("Test 1 (Empty String) -> OK 400")
else:
    print(f"Test 1 Failed -> Status: {response.status_code}")
    print(response.content[:200])

# Test 2: Spam message
spam_payload = {"text": "Winner! You have won a 1000 dollar prize. Click here."}
response = client.post('/api/v1/classify/', content_type="application/json", data=json.dumps(spam_payload))
if response.status_code == 200:
    print("Test 2 (Spam Message) -> OK 200:", response.json())
else:
    print(f"Test 2 Failed -> Status: {response.status_code}")
    print(response.content[:200])

# Test 3: Ham message
ham_payload = {"text": "Hey, how are you doing today?"}
response = client.post('/api/v1/classify/', content_type="application/json", data=json.dumps(ham_payload))
if response.status_code == 200:
    print("Test 3 (Ham Message)  -> OK 200:", response.json())
else:
    print(f"Test 3 Failed -> Status: {response.status_code}")
    print(response.content[:200])
