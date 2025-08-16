import requests

URL = "http://127.0.0.1:8000/"
ENDPOINT = "menu"

response = requests.get(URL + ENDPOINT)
print(response.json())


ENDPOINT = "orders"

body = {
    "customer_name": "John Doe",
    "items": ["հավով", "տավարով"],
    "special_requests": "Խնդրում եմ առանց սոխի"
}

response = requests.post(URL + ENDPOINT, json=body)
print(response.json())