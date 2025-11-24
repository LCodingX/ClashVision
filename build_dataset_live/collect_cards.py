import requests
import json

# === CONFIGURATION ===
API_URL = "https://api.clashroyale.com/v1/cards"
BEARER_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6ImQ4NjYxNGMwLTIxOGUtNDc4NC1hNmY5LTY3MTVjNWQ2ZDQwNyIsImlhdCI6MTc1MTA0NjE1NSwic3ViIjoiZGV2ZWxvcGVyLzQ5ZTYyNzFmLWQ2ZDYtNjg5OC1mNmYxLWQ0Njc3MWFlYzhjNyIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZlciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyIyMy4yNDIuMTc1LjE5NCJdLCJ0eXBlIjoiY2xpZW50In1dfQ.RIDK7j3RSlifXpScJu5J_yuz-Oyd_s56AJWEyJIzLXVzlkOUxd43R3jXbGmm0aR03iLDoV1O8emFpunK47MI9Q"  # Replace this with your actual token
OUTPUT_FILE = "clash_cards.json"

# === MAKE THE REQUEST ===
headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive"
}

response = requests.get(API_URL, headers=headers)
response.raise_for_status()

# === EXTRACT DESIRED FIELDS ===
cards_data = response.json().get("items", [])
filtered_cards = [
    {
        "name": card.get("name"),
        "elixirCost": card.get("elixirCost"),
        "iconUrls": card.get("iconUrls")
    }
    for card in cards_data
]

# === SAVE TO JSON FILE ===
with open(OUTPUT_FILE, "w") as f:
    json.dump(filtered_cards, f, indent=2)

print(f"✅ Saved {len(filtered_cards)} cards to {OUTPUT_FILE}")
