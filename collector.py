import os
import re
import sqlite3
import time
from datetime import datetime, timezone
import praw
from transformers import pipeline

DB_PATH = "cloudflare_pulse.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS reddit_items (
            id TEXT PRIMARY KEY,
            timestamp DATETIME,
            subreddit TEXT,
            text TEXT,
            sentiment TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

print("Loading Hugging Face sentiment model...")
hf_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True,
    max_length=512
)

reddit = praw.Reddit(
    client_id="yH0aTnJEt6qUgGn835B4vg",
    client_secret="",
    user_agent="org.quantumbadger.redreader/1.26"
)

TARGET_SUBREDDITS = "cloudflare+sysadmin+webdev"

def clean_text(text: str) -> str:
    # Drop URLs and markdown clutter; keep words and punctuation intact
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[\r\n]+", " ", text)
    return text.strip()

def process_and_save(item_id, created_utc, subreddit, raw_text):
    text = clean_text(raw_text)
    if len(text) < 15:  # Skip trivial comments like "same", "lol"
        return
    
    # Filter for relevance if coming from general sysadmin/webdev subreddits
    if subreddit.lower() != "cloudflare" and "cloudflare" not in text.lower():
        return

    # Run transformer
    result = hf_model(text)[0]
    sentiment_label = result["label"].lower()  # 'positive', 'neutral', 'negative'
    confidence = float(result["score"])
    
    dt = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("""
            INSERT OR IGNORE INTO reddit_items (id, timestamp, subreddit, text, sentiment, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (item_id, dt, subreddit, text, sentiment_label, confidence))
        conn.commit()
        print(f"[{dt}] [{sentiment_label.upper()} ({confidence:.2f})] {text[:60]}...")
    except Exception as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

def fetch_historical_backlog(subreddit, limit=100):
    """Fetches recent items on startup so you don't start with an empty database."""
    print(f"Bootstrapping: Fetching up to {limit} recent posts & comments...")
    count = 0
    
    for post in subreddit.new(limit=limit):
        full_text = f"{post.title}. {post.selftext}"
        process_and_save(f"post_{post.id}", post.created_utc, post.subreddit.display_name, full_text)
        count += 1

    for comment in subreddit.comments(limit=limit):
        process_and_save(f"comm_{comment.id}", comment.created_utc, comment.subreddit.display_name, comment.body)
        count += 1

    print(f"Initial backfill complete. Processed {count} items.")

def main():
    init_db()
    subreddit = reddit.subreddit(TARGET_SUBREDDITS)
    
    fetch_historical_backlog(subreddit, limit=100)

    print("\nListening for brand-new live items (heartbeat logs every 30s)...")
    last_heartbeat = time.time()
    
    while True:
        try:
            # Check comment stream
            for comment in subreddit.stream.comments(skip_existing=True, pause_after=5):
                if comment is None:
                    break
                process_and_save(f"comm_{comment.id}", comment.created_utc, comment.subreddit.display_name, comment.body)

            # Check submission stream
            for post in subreddit.stream.submissions(skip_existing=True, pause_after=5):
                if post is None:
                    break
                full_text = f"{post.title}. {post.selftext}"
                process_and_save(f"post_{post.id}", post.created_utc, post.subreddit.display_name, full_text)

            # Heartbeat printout so you know it's not frozen
            if time.time() - last_heartbeat > 30:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Polling... still connected and waiting for new mentions.")
                last_heartbeat = time.time()

        except Exception as e:
            print(f"Network/Stream error: {e}. Reconnecting in 10s...")
            time.sleep(10)

if __name__ == "__main__":
    main()