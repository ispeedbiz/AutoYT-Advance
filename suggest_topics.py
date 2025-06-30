# pyright: reportMissingImports=false
""" Suggest Topics Script for Back to Zero YouTube Channel """
from dotenv import load_dotenv
load_dotenv()

import os
import openai
import gspread
from openai import OpenAI
from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials
from utils import load_env

# Load environment variables (OpenAI API key, etc.)
env = load_env()

# Google Sheets setup
SERVICE_ACCOUNT_FILE = "service_account.json"
SHEET_NAME = "Back to Zero – Input Sheet"
WORKSHEET_INDEX = 0

# Configuration for ideas generation
NUM_DAYS = 10
IDEAS_PER_DAY = 2
TOTAL_IDEAS = NUM_DAYS * IDEAS_PER_DAY
LANGUAGE = "Hindi"
MORNING_TIME = "09:00"
EVENING_TIME = "18:00"

# Initialize OpenAI client
client = OpenAI(api_key=env["OPENAI_API_KEY"])

# Prompt for GPT-4 to generate video ideas
prompt = f"""
You are a strategist for a Hindi YouTube channel like 'Syllabus with Rohit'.
Suggest 20 *unique* and *emotional* video ideas in Hindi.

🔑 Rules:
• Custom Title = catchy, curiosity-driven, **in English only**, max 4 words
• Summary = 1–2 lines, Hindi with common English terms
• Key Highlights = 2–3 bullets (each ≤ 10 words, Hindi-English mix)
• Audience = who would watch this?
• Tone = Motivational, Informative, Storytelling etc.
• Duration = Short, Medium, or Long
• Thumbnail Source = PERSON / BOOK / AUTO
• Image URL (if applicable)

📋 Output Format:
Book/Topic:
Custom Title:
Summary:
Key Highlights:
- point 1
- point 2
- point 3 (optional)
Target Audience:
Style/Tone:
Duration:
Thumbnail Source:
Image URL:

Separate each idea with ===
"""

# Get ideas from GPT-4
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a Hindi content expert for YouTube."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    max_tokens=1800
)
raw = resp.choices[0].message.content
if raw is None:
    raise Exception("Failed to generate content from OpenAI")
raw = raw.strip()

# Split and parse the ideas
ideas = [i.strip() for i in raw.split("===") if i.strip()]
rows = []
seen_titles = set()

for i, idea in enumerate(ideas):
    lines = idea.split("\n")
    d = {"Language": LANGUAGE}
    key_map = {
        "book/topic": "Book/Topic",
        "custom title": "Custom Title (optional)",
        "summary": "Notes (optional)",
        "key highlights": "Key Highlights",
        "target audience": "Target Audience",
        "style/tone": "Style/Tone - E.g., Inspirational, Motivational, Storytelling, Informative, Humorous (used dynamically in prompts)",
        "duration": "Duration - Desired length of video (Short: 1-min, Medium: 5-min, Long: 10-min)",
        "thumbnail source": "Thumbnail Source (AUTO / BOOK / PERSON / NONE).",
        "image url": "book_img_url / person_img_url"
    }

    current_key = ""
    highlights = []
    for line in lines:
        line = line.strip().strip("*")  # remove any markdown bullets and whitespace
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip().lower()
            val = val.strip().strip("*")
            current_key = key
            mapped = key_map.get(key)
            if mapped:
                d[mapped] = val
        elif current_key == "key highlights" and line.startswith("-"):
            highlights.append(line.lstrip("-").strip())
    if highlights:
        d["Key Highlights"] = ", ".join(highlights)

    # Avoid duplicate ideas by title+topic
    title_key = d.get("Custom Title (optional)", "")[:60]
    topic_key = d.get("Book/Topic", "")[:60]
    unique_key = f"{title_key}|{topic_key}"
    if unique_key in seen_titles:
        continue
    seen_titles.add(unique_key)

    # Auto-fill Thumbnail Text if not provided (use title or topic)
    d["Thumbnail Text Specific catchy text to display on thumbnail"] = title_key or topic_key

    # Default CTA and empty attribution
    d["CTA (Call-to-Action) - What you want viewers to do after watching (Subscribe, comment, share, etc.)"] = "Subscribe for more!"
    d["Attribution Note"] = ""

    # Schedule publishing times (2 per day: morning & evening)
    day_offset = i // IDEAS_PER_DAY
    publish_time = MORNING_TIME if i % 2 == 0 else EVENING_TIME
    publish_date = (datetime.now() + timedelta(days=day_offset)).strftime("%Y-%m-%d")
    d["Publish Date"] = f"{publish_date} {publish_time}"
    d["Status"] = "🕓 Waiting"

    rows.append(d)

# Write results to Google Sheet
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
gc = gspread.authorize(creds)
ws = gc.open(SHEET_NAME).get_worksheet(WORKSHEET_INDEX)

header = ws.row_values(1)
for d in rows:
    row = [d.get(col, "") for col in header]
    ws.append_row(row, value_input_option="USER_ENTERED")

print(f"✅ {len(rows)} ideas written to Google Sheet '{SHEET_NAME}'.")
