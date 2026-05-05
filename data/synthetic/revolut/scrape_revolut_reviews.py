"""
Scrape real Revolut reviews from Google Play Store.

Output: data/synthetic/revolut/reviews_revolut.csv with the same schema
that Mengda used for Lidl Plus (reviewId, content, score, app).

App Store scraping was removed — the app-store-scraper package depends on
an old urllib3 that is incompatible with Python 3.13. Google Play alone
gives us 300+ reviews, which is plenty for the brief (20-30 minimum).

Run once:
    pip install google-play-scraper
    python3 scrape_revolut_reviews.py
"""

import csv
from pathlib import Path

from google_play_scraper import reviews, Sort

print("Scraping Google Play Store reviews for Revolut...")

gp_reviews, _ = reviews(
    "com.revolut.revolut",       # Revolut's package name on Google Play
    lang="en",
    country="us",
    sort=Sort.NEWEST,
    count=500,                    # ask for 500 most recent
)

print(f"  Got {len(gp_reviews)} Google Play reviews")


# --- Write to CSV ---
out_dir = Path(__file__).parent
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "reviews_revolut.csv"

written = 0
with out_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["reviewId", "content", "score", "app"])

    for r in gp_reviews:
        # Skip empty review bodies — they're noise.
        content = (r.get("content") or "").strip()
        if not content:
            continue
        writer.writerow([
            r.get("reviewId"),
            content,
            r.get("score"),
            "Revolut",
        ])
        written += 1

print(f"\nDone. Wrote {written} reviews to {out_path}")
