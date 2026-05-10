import requests
import json
import os

JSON_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
OUTPUT_FILE = "../data/shl_catalog.json"

def fetch_shl_catalog():
    print(f"Fetching data directly from {JSON_URL}...")
    
    response = requests.get(JSON_URL)
    if response.status_code != 200:
        print(f"Failed to retrieve JSON: {response.status_code}")
        return

    # --- THE FIX IS HERE ---
    try:
        # strict=False tells Python to ignore unescaped control characters
        raw_data = json.loads(response.text, strict=False)
    except json.JSONDecodeError as e:
        print("Standard parsing failed. Attempting aggressive text cleanup...")
        # If strict=False isn't enough, we manually scrub the raw text of bad hidden characters
        cleaned_text = response.text.replace('\r', '').replace('\n', '\\n').replace('\t', '\\t')
        raw_data = json.loads(cleaned_text, strict=False)
    
    print(f"\nSuccess! Found {len(raw_data)} total items.")
    print("Here is what the first item looks like:")
    print(json.dumps(raw_data[0], indent=2))
    
    # --- MAPPING PHASE ---
    # --- MAPPING PHASE ---
    processed_catalog = []
    
    for item in raw_data:
        # Safely get the keys array. If it exists and has items, take the first one. 
        # Otherwise, fallback to the word "Assessment".
        keys_array = item.get("keys", [])
        test_category = keys_array[0] if keys_array else "Assessment"
        
        processed_catalog.append({
            "name": item.get("name", "Unknown Name"), 
            "url": item.get("link", "No URL"), # Using the 'link' fix you discovered!
            "test_type": test_category, 
            "description": item.get("description", "")
        })

    # Save to our data folder
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_catalog, f, indent=4)
        
    print(f"\nData successfully formatted and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_shl_catalog()