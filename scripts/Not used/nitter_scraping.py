import csv
import time
import urllib.parse
import requests
from bs4 import BeautifulSoup

# --------------------
# CONFIG
# --------------------

# Pick a Nitter instance that is alive & that you’re allowed to scrape.
# You can swap this for another instance, e.g. "https://nitter.lacontrevoie.fr"
BASE_URL = "https://nitter.space"

# Twitter search query – Nitter forwards this to Twitter search
QUERY = "geotherm* OR aardwarmte*"

# Optional date filter (Twitter-style; can be empty strings if you don't care)
SINCE = "2024-01-02"        # e.g. "2024-01-01"
UNTIL = "2024-12-31"        # e.g. "2024-12-31"

# How many pages of search results to fetch
MAX_PAGES = 5

# Pause between requests (seconds) to be polite / reduce blocking risk
REQUEST_DELAY = 3

OUTPUT_CSV = "tweets_geotherm_aardwarmte.csv"

# --------------------
# SCRAPER
# --------------------


def build_search_url(page: int) -> str:
    """
    Build Nitter search URL for a given page.
    """
    params = {
        "f": "tweets",
        "q": QUERY,
    }
    if SINCE:
        params["since"] = SINCE
    if UNTIL:
        params["until"] = UNTIL
    if page > 1:
        params["p"] = str(page)

    return f"{BASE_URL}/search?{urllib.parse.urlencode(params)}"

def parse_tweets_from_html(html: str):
    """
    Parse tweets from a Nitter search results page.
    Returns a list of dicts.
    """
    soup = BeautifulSoup(html, "html.parser")
    tweets = []

    # Nitter tweet items live in elements with class="timeline-item"
    items = soup.select(".timeline-item")
    print(f"Found {len(items)} elements with class 'timeline-item'.")

    for article in items:
        try:
            # Username (e.g. <a class="username">@user</a>)
            user_el = article.select_one("a.username")
            username = user_el.get_text(strip=True).lstrip("@") if user_el else ""

            # Display name
            name_el = article.select_one("a.fullname")
            display_name = name_el.get_text(strip=True) if name_el else ""

            # Date + Tweet URL (tweet-date/tweet-link depending on layout)
            date_el = article.select_one("a.tweet-date, a.tweet-link")
            tweet_url = ""
            created_at = ""
            if date_el:
                href = date_el.get("href", "")
                tweet_url = urllib.parse.urljoin(BASE_URL, href)
                created_at = date_el.get("title", "").strip() or date_el.get_text(strip=True)

            # Tweet text
            content_el = article.select_one(".tweet-content")
            text = content_el.get_text(" ", strip=True) if content_el else ""

            # Retweets / replies / likes (if available)
            stats_el = article.select_one(".tweet-stats")
            replies = retweets = likes = ""
            if stats_el:
                # Nitter usually has spans with aria-labels or titles
                for span in stats_el.select("span"):
                    label = (span.get("aria-label") or span.get("title") or "").lower()
                    val = span.get_text(strip=True)
                    if "repl" in label:
                        replies = val
                    elif "retweet" in label:
                        retweets = val
                    elif "like" in label:
                        likes = val

            tweets.append(
                {
                    "username": username,
                    "display_name": display_name,
                    "created_at": created_at,
                    "text": text,
                    "tweet_url": tweet_url,
                    "replies": replies,
                    "retweets": retweets,
                    "likes": likes,
                }
            )
        except Exception as e:
            print("Error parsing one tweet:", e)
            continue
    
    if not tweets:
        with open("last_page.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("No tweets parsed; saved raw HTML to last_page.html for inspection.")
        
    return tweets


def fetch_page(page: int):
    """
    Fetch one page of Nitter search results.
    """
    url = build_search_url(page)
    print(f"Fetching page {page}: {url}")

    headers = {
        # Try to look as much like a normal browser as possible
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://nitter.net/",
    }

    resp = requests.get(url, headers=headers, timeout=30)
    print("Status code:", resp.status_code)
    print("Response length:", len(resp.text))

    # Save raw response for inspection
    with open("last_page.html", "w", encoding="utf-8") as f:
        f.write(resp.text)

    resp.raise_for_status()

    if not resp.text.strip():
        raise RuntimeError(
            "Got an empty response body from Nitter. "
            "This instance may be blocking non-browser clients."
        )

    return resp.text



def main():
    all_tweets = []

    for page in range(1, MAX_PAGES + 1):
        try:
            html = fetch_page(page)
        except requests.HTTPError as e:
            print("HTTP error:", e)
            break
        except requests.RequestException as e:
            print("Request error:", e)
            break

        tweets = parse_tweets_from_html(html)
        if not tweets:
            print("No tweets found on this page; stopping.")
            break

        print(f"  Parsed {len(tweets)} tweets from page {page}.")
        all_tweets.extend(tweets)

        time.sleep(REQUEST_DELAY)

    if not all_tweets:
        print("No tweets collected.")
        return

    # Write CSV
    fieldnames = [
        "username",
        "display_name",
        "created_at",
        "text",
        "tweet_url",
        "replies",
        "retweets",
        "likes",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in all_tweets:
            writer.writerow(t)

    print(f"Saved {len(all_tweets)} tweets to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
