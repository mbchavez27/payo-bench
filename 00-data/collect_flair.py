"""Collect post flair for the bronze r/adviceph dump.

The original dump has no flair field, so this script re-queries Reddit's official
API (`/api/info`) for every post ID and writes the flair as a new, separate bronze
folder. The original bronze files are never touched.

Reddit blocks unauthenticated requests, so this needs a Reddit "script" app
(https://www.reddit.com/prefs/apps). Put its credentials in `.env`
(see `.env.example`).

Usage:
    uv run python 00-data/collect_flair.py
"""

import hashlib
import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

BRONZE_DIR = Path(__file__).resolve().parent / "bronze"
SOURCE_POSTS = BRONZE_DIR / "reddit-adviceph_2026-11-06" / "posts.jsonl"
TODAY = date.today().isoformat()
OUT_DIR = BRONZE_DIR / f"reddit-adviceph-flair_{TODAY}"
OUT_FILE = OUT_DIR / "flair.jsonl"
MANIFEST = BRONZE_DIR / "MANIFEST.json"

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
ENDPOINT = "https://oauth.reddit.com/api/info"
BATCH_SIZE = 100
SLEEP_SECONDS = 2.0
MAX_RETRIES = 5
HEADERS = {"User-Agent": "payo-bench-research/0.1"}
FIELDS = ("link_flair_text", "link_flair_template_id")


def load_post_ids(path: Path) -> list[str]:
    """
    Read post IDs from the bronze posts file, keeping file order.

    Args:
        path: Path to posts.jsonl

    Returns:
        List of Reddit post IDs (without the t3_ prefix)
    """
    with path.open(encoding="utf-8") as f:
        return [json.loads(line)["id"] for line in f]


def get_token() -> str:
    """
    Get an app-only OAuth token using the client credentials in `.env`.

    Returns:
        Bearer access token
    """
    load_dotenv()
    auth = (os.environ["REDDIT_CLIENT_ID"], os.environ["REDDIT_CLIENT_SECRET"])
    resp = requests.post(TOKEN_URL, auth=auth, data={"grant_type": "client_credentials"}, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_batch(ids: list[str], token: str) -> list[dict]:
    """
    Fetch flair for up to 100 posts, retrying on rate limits and server errors.

    Args:
        ids: Reddit post IDs
        token: OAuth bearer token

    Returns:
        One record per post Reddit returned, with id, flair fields, and retrieval time
    """
    url = ENDPOINT
    params = {"id": ",".join(f"t3_{i}" for i in ids)}
    headers = {**HEADERS, "Authorization": f"bearer {token}"}
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 200:
            break
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(SLEEP_SECONDS * 2 ** (attempt + 1))
            continue
        resp.raise_for_status()
    else:
        raise RuntimeError(f"Gave up after {MAX_RETRIES} retries: {url}")

    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return [
        {"id": child["data"]["id"], **{k: child["data"].get(k) for k in FIELDS}, "retrieved_at": retrieved_at}
        for child in resp.json()["data"]["children"]
    ]


def file_entry(path: Path) -> dict:
    """
    Build a manifest entry for one file.

    Args:
        path: File to describe

    Returns:
        Dict with sha256, row count, and size in bytes
    """
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "rows": data.count(b"\n"), "bytes": len(data)}


def main() -> None:
    """Fetch flair for every bronze post and record the new file in the manifest."""
    if OUT_FILE.exists():
        raise FileExistsError(f"{OUT_FILE} already exists; bronze is immutable")

    ids = load_post_ids(SOURCE_POSTS)
    token = get_token()
    records: list[dict] = []
    for start in range(0, len(ids), BATCH_SIZE):
        records.extend(fetch_batch(ids[start : start + BATCH_SIZE], token))
        print(f"{min(start + BATCH_SIZE, len(ids))}/{len(ids)}")
        time.sleep(SLEEP_SECONDS)

    missing = set(ids) - {r["id"] for r in records}
    print(f"Fetched {len(records)} posts; {len(missing)} not returned by Reddit")

    OUT_DIR.mkdir(parents=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT_DIR / ".gitkeep").touch()

    manifest = json.loads(MANIFEST.read_text())
    manifest["files"][str(OUT_FILE.relative_to(BRONZE_DIR))] = {
        **file_entry(OUT_FILE),
        "collected_at": TODAY,
        "source": "Reddit OAuth API /api/info",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
