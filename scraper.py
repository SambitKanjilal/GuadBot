import json
import os

import psycopg2
from cachetools import LRUCache
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

load_dotenv()

# In-memory LRU: evicts least-recently-used part when 128-part limit is reached
_cache: LRUCache = LRUCache(maxsize=128)

_conn = psycopg2.connect(os.getenv("POSTGRES_DSN", "postgresql://postgres:password@localhost:5432/parts"))

with _conn.cursor() as _cur:
    _cur.execute("""
        CREATE TABLE IF NOT EXISTS scraped_parts (
            part_name   TEXT PRIMARY KEY,
            mouser_data TEXT,
            digikey_data TEXT,
            scraped_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)
_conn.commit()


def _db_lookup(part_name: str) -> dict | None:
    with _conn.cursor() as cur:
        cur.execute("SELECT mouser_data, digikey_data FROM scraped_parts WHERE part_name = %s", (part_name,))
        row = cur.fetchone()
    return {"mouser": row[0], "digikey": row[1]} if row else None


def _db_save(part_name: str, data: dict):
    with _conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO scraped_parts (part_name, mouser_data, digikey_data)
            VALUES (%s, %s, %s)
            ON CONFLICT (part_name) DO UPDATE
            SET mouser_data = EXCLUDED.mouser_data,
                digikey_data = EXCLUDED.digikey_data,
                scraped_at = NOW()
            """,
            (part_name, data["mouser"], data["digikey"]),
        )
    _conn.commit()


def _scrape(part_name: str) -> dict:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        page.goto(f"https://www.mouser.com/c/?q={part_name}")
        page.wait_for_load_state("networkidle")
        mouser_text = page.inner_text("body")[:3000]

        page.goto(f"https://www.digikey.com/en/products/result?keywords={part_name}")
        page.wait_for_load_state("networkidle")
        digikey_text = page.inner_text("body")[:3000]

        browser.close()

    return {"mouser": mouser_text, "digikey": digikey_text}


def get_part_info(part_name: str) -> dict:
    if part_name in _cache:
        return _cache[part_name]

    result = _db_lookup(part_name)
    if result is None:
        result = _scrape(part_name)
        _db_save(part_name, result)

    _cache[part_name] = result
    return result
