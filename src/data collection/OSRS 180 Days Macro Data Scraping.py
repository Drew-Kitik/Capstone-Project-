# -*- coding: utf-8 -*-

# Code for the 'Macro' dataset that requests the farthest publically available data (180 days at 24H timepoints)

# Features
# 24H timeseries with volume for a maximum of 180 days data per item if available. 
# Retries + exponential backoff + jitter. Mostly was experimental workarounds/experimentation before realizing API had a cooldown pattern.
# Implemented periodic cooldowns for requests: API has a pull cooldown that occured every 2-3 minutes of continued requests (regardless of delay).
# Incremental saving every N items: Due to the sheer size of requests this was important to avoid having to restart completly from the top of the list.
# Resume from checkpoint feature: skip already-processed items if situation above occurs. "
import os
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from json import JSONDecodeError
import random

# --------------------
# Config
# --------------------
# No matter how fine tuned these settings were without the cooldown feature the API pull would fail regardless of the request delay lengths.
# API would get upset roughly after 90-120 seconds of continous requests, implemented cooldown period (fine tuned via experimentation)
# Index range refers to position of the sorted mapping list (alphabetical) and is not the raw API order for the MAPPING URL link below. 
# Features below are for custom set values for delays between requests, set cooldown per x requests, cooldown length and incremental save length. 
REQUEST_DELAY = 2                   
COOLDOWN_EVERY_N_REQUESTS = 200     
COOLDOWN_SECONDS = 60               
INCREMENTAL_SAVE_EVERY_ITEMS = 200  


# Project root adjusted for Github (two levels up if inside src/data_collection/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "macro"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARTIAL_PATH = OUTPUT_DIR / "osrs_macro_timeseries_partial.parquet"
CHECKPOINT_PATH = OUTPUT_DIR / "macro_checkpoint.txt"


# Per best practices information for the API's usage they required a header for any API access along with contact information. 
# If running on your own from github repository, please replace with your own contact information.
HEADERS = {
    "User-Agent": "OSRS API Scrapper - Contact: https://github.com/drew-kitik"
}
# Mapping/Timeseries URL provided by API owners which match up the records were are seeking to a number index (hence start/end INDEX variables)
MAPPING_URL = "https://prices.runescape.wiki/api/v1/osrs/mapping"
GRAPH_URL   = "https://secure.runescape.com/m=itemdb_oldschool/api/graph/{}.json"

# --------------------
# Setup
# --------------------
# Line below is a check to make sure the output folder exists before starting the pull.
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Variable for performing the API data pull and pulling in the HEADER variable. 
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# --------------------
# Helpers
# --------------------
def get_json_with_retries(url, params=None, max_retries=4, base_sleep=3):
#   This section was made from troubleshooting the API failing to return data after a certain point (later found to be 90-120 seconds of constant requests)
#    Features:
#    Sends HTTP request for a JSON endpoint (Request.Session) with exponential backoffs with random jitters.
#    Automatic retries on the following errors: Rate limit (429), server 5xx, network errors/hiccups, empty body, invalid JSON etc.
#    Exponential Backoff Formula: base * 2^(attempt-1) * jitter
    attempt = 0
    while True:
        try:
            resp = SESSION.get(url, params=params, timeout=15)
            status = resp.status_code
            #Retry on HTTP status, 429 error = rate-limited, 5xx = server error.
            if status == 429 or 500 <= status < 600:
                attempt += 1
                if attempt > max_retries:
                    print(f"Giving up on {url} after {max_retries} retries (HTTP {status}).")
                    return None
                #Expontential backoff with jitter (10~30%). 
                sleep_s = base_sleep * (2 ** (attempt - 1)) * random.uniform(0.9, 1.3)
                print(f"HTTP {status} for {url}. Retry {attempt}/{max_retries} in {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue
            # Custom retries for 429 or 5xx (rate limit/server error) otherwise raise_for_status() for any non-2xx response. 
            resp.raise_for_status()

            if not resp.content:
                attempt += 1
                if attempt > max_retries:
                    print(f"Empty response for {url}. Gave up after {max_retries} retries.")
                    return None
                sleep_s = base_sleep * (2 ** (attempt - 1)) * random.uniform(0.9, 1.3)
                print(f"Empty response for {url}. Retry {attempt}/{max_retries} in {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue
            # JSON parse if body isn't a valid JSON output
            try:
                return resp.json()
            except JSONDecodeError:
                attempt += 1
                if attempt > max_retries:
                    print(f"Invalid JSON for {url}. Gave up after {max_retries} retries.")
                    return None
                sleep_s = base_sleep * (2 ** (attempt - 1)) * random.uniform(0.9, 1.3)
                print(f"Invalid JSON for {url}. Retry {attempt}/{max_retries} in {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue
            # Retry network layer failures (connection issues/DNS/timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            attempt += 1
            if attempt > max_retries:
                print(f"Network error for {url}: {e}. Gave up after {max_retries} retries.")
                return None
            sleep_s = base_sleep * (2 ** (attempt - 1)) * random.uniform(0.9, 1.3)
            print(f"Network error for {url}: {e}. Retry {attempt}/{max_retries} in {sleep_s:.1f}s...")
            time.sleep(sleep_s)
            continue
        # Non-retryable requests expections (bad requests, invalid URL or others)
        except requests.RequestException as e:
            print(f"Request failed for {url}: {e}")
            return None
# Ready a new line delimited file of processed item IDs and return them as 0/1 for membership check, skips previous runs.
def load_checkpoint_ids(path):
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                ids.add(int(line))
    return ids
# Puts single processed item ID number into checkpointfile, append as the file runs so it saves as we go (incredibly important feature to have!)
def append_checkpoint_id(path, item_id):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{item_id}\n")
# Fetches the full bulk item list from the MAPPING URL used with the GET helper variable, returns the structured output desired. 
# Includes sleep timer for failed gets, keeps only the fields we want and uses lambda sort list for logs to be easier to process/read.
def fetch_mapping():
    """Fetch full OSRS item mapping once (id, name, members, etc.)."""
    data = get_json_with_retries(MAPPING_URL, params=None, max_retries=4, base_sleep=3)
    if data is None:
        raise RuntimeError("Failed to fetch /mapping data after retries.")
    # mapping returns a list of dicts; we only need id + name for macro
    # Keep sort for stable processing order
    items = [{"id": int(d["id"]), "name": d.get("name", "Unknown")} for d in data if "id" in d]
    items.sort(key=lambda x: (x["name"].lower(), x["id"]))
    print(f"Mapping returned {len(items)} items.")
    return items
# This function does the actual fetching of the 24 hour timepoint series (180 days) for a single item ID additionally:
# Wraps the request with retries and logs both success and failure cases
# If no 'daily' data is returned (untradeable item, newly released item with no history etc)
# The function returns an empty dict variable instead of failing
# Includes arguement if no legitimate data in 24H timepoint (untradable items, newly implemented item (missing price history) etc) 
def fetch_timeseries_data(item_id):
    """Fetch /graph/{id}.json and log success/failure clearly."""
    url = GRAPH_URL.format(item_id)
    data = get_json_with_retries(url, params=None, max_retries=4, base_sleep=3)
    if data is None:
        print(f"  -> Final: request failed for item {item_id}. Skipping.")
        return {}
    daily = data.get("daily", {})
    if not daily:
        print(f"  -> Final: no 'daily' data available for item {item_id}. Skipping.")
        return {}
    print(f"  -> Success: received {len(daily)} daily points for item {item_id}.")
    return data

# --------------------
# Main
# --------------------
all_rows = []
processed_ids = load_checkpoint_ids(CHECKPOINT_PATH)
print(f"Loaded {len(processed_ids)} processed item IDs from checkpoint.")

# Fetches all the item IDs from the MAPPING URL variable.
all_items = fetch_mapping()

requests_since_cooldown = 0
items_since_save = 0

for idx, item in enumerate(all_items, start=1):
    item_id = item["id"]
    name = item["name"]

    if item_id in processed_ids:
        continue

    print(f"[{idx}/{len(all_items)}] Fetching 180-day timeseries for item: {name} (ID: {item_id})")
    graph_data = fetch_timeseries_data(item_id)

# Countdown output for cooldown
    requests_since_cooldown += 1
    if requests_since_cooldown >= COOLDOWN_EVERY_N_REQUESTS:
        print(f"== Cooldown: reached {COOLDOWN_EVERY_N_REQUESTS} graph requests. Sleeping {COOLDOWN_SECONDS}s ==")
        time.sleep(COOLDOWN_SECONDS)
        requests_since_cooldown = 0

    time.sleep(REQUEST_DELAY)

    daily_prices = graph_data.get("daily", {})
    if daily_prices:
        for ts, avg_price in daily_prices.items():
            all_rows.append({
                "Item ID": item_id,
                "Item Name": name,
                "Date": pd.to_datetime(int(ts), unit='ms'),
                "Average Price": avg_price
            })

# This section made to prevent looping forever on bad/missing items IDs. 
    processed_ids.add(item_id)
    append_checkpoint_id(CHECKPOINT_PATH, item_id)
    items_since_save += 1

# Incremental saving feature.
    if items_since_save >= INCREMENTAL_SAVE_EVERY_ITEMS:
        print(f"-- Incremental save after {INCREMENTAL_SAVE_EVERY_ITEMS} items --")
        pd.DataFrame(all_rows).to_parquet(PARTIAL_PATH, index=False)
        items_since_save = 0

# Final save. 
today_str = datetime.today().strftime("%Y-%m-%d")
final_path = os.path.join(OUTPUT_DIR, f"osrs_macro_timeseries_AZ_{today_str}.parquet")
pd.DataFrame(all_rows).to_parquet(final_path, index=False)
print(f"Macro dataset saved as '{final_path}'")
print(f"Checkpoint file at: {CHECKPOINT_PATH}")
if os.path.exists(PARTIAL_PATH):
    print(f"Latest partial file at: {PARTIAL_PATH}")

