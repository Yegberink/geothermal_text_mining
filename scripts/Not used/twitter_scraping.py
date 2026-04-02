import asyncio
import csv
import random
from datetime import date, timedelta
from twikit import Client

# --- CONFIG ---
USERNAME_OR_EMAIL_1 = "YannickE52068"
USERNAME_OR_EMAIL_2 = "yannickegberink@gmail.com"   # optional but recommended
PASSWORD = "Twitterja12"    

OUT_CSV = "tweets_2025_geothermal_aardwarmte.csv"

# Add/adjust variants here (since wildcard * isn't reliable in X search)
TERMS = [
    "geothermie",
    "aardwarmte",
    "aardwarmtebron",
    "aardwarmteproject",
    "aardwarmteput",
    "geothermisch",
]

LANG_FILTER = "lang:nl"
EXCLUDE_RETWEETS = True  # set False if you want them

# Slice size: 1 day is safest; 7 days is faster but may miss more in very busy periods
SLICE_DAYS = 1

# --- HELPERS ---
def build_base_query() -> str:
    terms_q = " OR ".join(TERMS)
    q = f"({terms_q}) {LANG_FILTER}"
    if EXCLUDE_RETWEETS:
        q += " -filter:retweets"
    return q

async def paginate_search(client: Client, query: str, product: str = "Latest", count: int = 20):
    """
    Yields tweets for a query using twikit pagination.
    """
    res = await client.search_tweet(query, product, count=count)
    while res and not res.empty():
        for t in res:
            yield t
        # gentle jitter to reduce hammering
        await asyncio.sleep(random.uniform(0.4, 1.2))
        res = await res.next()

def safe_text(t) -> str:
    return (getattr(t, "full_text", None) or getattr(t, "text", "") or "").replace("\n", " ").strip()

def safe_user(t):
    u = getattr(t, "user", None)
    if not u:
        return ("", "", "")
    return (
        getattr(u, "id", "") or "",
        getattr(u, "name", "") or "",
        getattr(u, "screen_name", "") or getattr(u, "username", "") or "",
    )

async def main():
    client = Client(language="nl-NL")

    # cookies_file lets you reuse login without re-entering credentials each run
    await client.login(
        auth_info_1=USERNAME_OR_EMAIL_1,
        auth_info_2=USERNAME_OR_EMAIL_2,
        password=PASSWORD,
        cookies_file="cookies.json",
    )

    base = build_base_query()

    start = date(2025, 1, 1)
    end = date(2026, 1, 1)  # exclusive
    step = timedelta(days=SLICE_DAYS)

    seen_ids = set()

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "tweet_id", "created_at",
            "user_id", "user_name", "user_handle",
            "text", "url",
            "reply_count", "retweet_count", "like_count", "quote_count",
            "query_since", "query_until",
        ])

        d = start
        while d < end:
            d2 = min(d + step, end)
            q = f'{base} since:{d.isoformat()} until:{d2.isoformat()}'

            print(f"Searching {d} → {d2} ...")
            try:
                async for t in paginate_search(client, q, product="Latest", count=20):
                    tid = getattr(t, "id", None)
                    if not tid or tid in seen_ids:
                        continue
                    seen_ids.add(tid)

                    created_at = getattr(t, "created_at", "") or ""
                    text = safe_text(t)
                    user_id, user_name, user_handle = safe_user(t)

                    # URL is best-effort; twikit fields can vary by version
                    url = getattr(t, "url", "") or ""
                    if not url and user_handle and tid:
                        url = f"https://x.com/{user_handle}/status/{tid}"

                    # counts are best-effort (field names vary)
                    reply_count = getattr(t, "reply_count", "") or ""
                    retweet_count = getattr(t, "retweet_count", "") or ""
                    like_count = getattr(t, "favorite_count", "") or getattr(t, "like_count", "") or ""
                    quote_count = getattr(t, "quote_count", "") or ""

                    w.writerow([
                        tid, created_at,
                        user_id, user_name, user_handle,
                        text, url,
                        reply_count, retweet_count, like_count, quote_count,
                        d.isoformat(), d2.isoformat(),
                    ])

            except Exception as e:
                # If X throttles / transient failure, wait and continue.
                print(f"  Error on slice {d}→{d2}: {e}")
                await asyncio.sleep(random.uniform(10, 20))

            d = d2
            # pause between slices
            await asyncio.sleep(random.uniform(1.0, 2.5))

    print(f"Done. Wrote {len(seen_ids)} unique tweets to {OUT_CSV}")

if __name__ == "__main__":
    asyncio.run(main())
