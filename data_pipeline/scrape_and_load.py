"""
Zepto Capstone - Module 1: Data Pipeline
Scrapes books.toscrape.com, cleans data, converts currency,
loads into a normalized SQLite DB, and runs SQL + pandas queries.

Run: python scrape_and_load.py
"""

import re
import sqlite3
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "http://books.toscrape.com/"
CATALOGUE_URL = "http://books.toscrape.com/catalogue/"
GBP_TO_INR = 105.50  # Fixed project-defined baseline rate (no date reference needed)

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

# ---------------------------------------------------------------------------
# STEP 1: SCRAPE
# ---------------------------------------------------------------------------

def get_category_links(n_categories=3):
    """Get links for the first n_categories from the sidebar."""
    resp = requests.get(BASE_URL, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    links = soup.select("div.side_categories ul li ul li a")
    categories = []
    for a in links[:n_categories]:
        name = a.text.strip()
        href = BASE_URL + a["href"]
        categories.append((name, href))
    return categories


def scrape_category(category_name, url):
    """Scrape all books in a category (handles pagination)."""
    books = []
    page_url = url
    while page_url:
        resp = requests.get(page_url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        articles = soup.select("article.product_pod")

        for art in articles:
            title = art.h3.a["title"].strip()
            price_text = art.select_one("p.price_color").text.strip()
            rating_class = art.select_one("p.star-rating")["class"]
            rating_word = [c for c in rating_class if c != "star-rating"][0]
            availability = art.select_one("p.instock.availability").text.strip()

            books.append({
                "title": title,
                "price": price_text,
                "star_rating": rating_word,
                "availability": availability,
                "category": category_name,
            })

        # pagination
        next_btn = soup.select_one("li.next a")
        if next_btn:
            page_url = page_url.rsplit("/", 1)[0] + "/" + next_btn["href"]
        else:
            page_url = None
        time.sleep(0.3)  # be polite to the practice server
    return books


def scrape_all(min_books=60, n_categories=3):
    categories = get_category_links(n_categories)
    all_books = []
    for name, url in categories:
        all_books.extend(scrape_category(name, url))

    # If not enough books from 3 categories, add more categories until threshold met
    idx = n_categories
    all_cats = get_category_links(50)
    while len(all_books) < min_books and idx < len(all_cats):
        name, url = all_cats[idx]
        all_books.extend(scrape_category(name, url))
        idx += 1

    print(f"Scraped {len(all_books)} books across categories.")
    return pd.DataFrame(all_books)


# ---------------------------------------------------------------------------
# STEP 2: CLEAN
# ---------------------------------------------------------------------------

def clean_data(df):
    df = df.copy()

    # price_gbp: strip currency symbol -> float
    df["price_gbp"] = (
        df["price"].str.replace(r"[^\d.]", "", regex=True).astype(float)
    )

    # rating: word -> int
    df["rating"] = df["star_rating"].map(RATING_MAP)

    # in_stock: text -> bool
    df["in_stock"] = df["availability"].str.contains("In stock", case=False, na=False)

    # Handle rows that failed to parse
    before = len(df)
    numeric_fail = df["price_gbp"].isna() | df["rating"].isna()
    if numeric_fail.any():
        # median-imputation for numeric fields where possible, else drop
        median_price = df["price_gbp"].median()
        median_rating = df["rating"].median()
        df.loc[df["price_gbp"].isna(), "price_gbp"] = median_price
        df.loc[df["rating"].isna(), "rating"] = median_rating
    df = df.dropna(subset=["title", "in_stock"])
    after = len(df)
    print(f"Cleaning: {before} -> {after} rows (dropped rows with unparseable "
          f"title/availability; numeric fields median-imputed where needed).")

    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    # STEP 3: currency conversion (fixed baseline rate)
    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]


# ---------------------------------------------------------------------------
# STEP 4: LOAD INTO NORMALIZED SQLITE SCHEMA
# ---------------------------------------------------------------------------

def build_database(df, db_path="zepto_books.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
    DROP TABLE IF EXISTS books;
    DROP TABLE IF EXISTS categories;

    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE
    );

    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        price_gbp REAL,
        price_inr REAL,
        rating INTEGER,
        in_stock INTEGER,
        category_id INTEGER REFERENCES categories(category_id)
    );
    """)
    conn.commit()

    # insert categories
    cat_names = df["category"].unique().tolist()
    cur.executemany(
        "INSERT INTO categories (category_name) VALUES (?)",
        [(c,) for c in cat_names],
    )
    conn.commit()

    cat_map = dict(pd.read_sql("SELECT category_id, category_name FROM categories", conn)
                   .values[:, ::-1])

    # insert books
    rows = [
        (r.title, r.price_gbp, r.price_inr, r.rating, int(r.in_stock), cat_map[r.category])
        for r in df.itertuples(index=False)
    ]
    cur.executemany(
        """INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# STEP 5: SQL QUERIES
# ---------------------------------------------------------------------------

def run_queries(conn, log_path="query_log.txt"):
    log_lines = []
    queries = {
        "Q1_select_where": """
            SELECT title, price_inr, rating FROM books
            WHERE in_stock = 1
            LIMIT 5;
        """,
        "Q2_order_by_limit": """
            SELECT title, price_inr FROM books
            ORDER BY price_inr DESC
            LIMIT 10;
        """,
        "Q3_distinct": """
            SELECT DISTINCT rating FROM books
            ORDER BY rating;
        """,
        "Q4_between_in": """
            SELECT title, price_inr FROM books
            WHERE price_inr BETWEEN 500 AND 2000
              AND rating IN (4, 5)
            LIMIT 10;
        """,
        "Q5_join_top_rated_per_category": """
            SELECT c.category_name, b.title, b.rating, b.price_inr
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            WHERE b.rating = 5
            ORDER BY c.category_name, b.price_inr DESC
            LIMIT 10;
        """,
    }

    results = {}
    for name, q in queries.items():
        df_result = pd.read_sql(q, conn)
        results[name] = df_result
        header = f"\n--- {name} ---\nSQL:\n{q.strip()}\n\nOutput:"
        body = df_result.to_string(index=False)
        print(header)
        print(body)
        log_lines.append(header)
        log_lines.append(body)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nAll query text + output also saved to {log_path}")
    return results


def compare_read_sql_vs_merge(conn):
    """Reproduce the JOIN query using pd.merge instead of SQL."""
    books_df = pd.read_sql("SELECT * FROM books", conn)
    cats_df = pd.read_sql("SELECT * FROM categories", conn)

    sql_join = pd.read_sql("""
        SELECT c.category_name, b.title, b.rating, b.price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.rating = 5
        ORDER BY c.category_name, b.price_inr DESC
        LIMIT 10;
    """, conn)

    merged = books_df.merge(cats_df, on="category_id")
    pandas_merge = (
        merged[merged["rating"] == 5][["category_name", "title", "rating", "price_inr"]]
        .sort_values(["category_name", "price_inr"], ascending=[True, False])
        .head(10)
        .reset_index(drop=True)
    )

    print("\n--- pd.read_sql result ---")
    print(sql_join.reset_index(drop=True).to_string(index=False))
    print("\n--- pd.merge result ---")
    print(pandas_merge.to_string(index=False))

    return sql_join, pandas_merge


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Scraping books.toscrape.com ...")
    raw_df = scrape_all(min_books=60, n_categories=3)
    raw_df.to_csv("raw_books.csv", index=False)

    print("\nCleaning data ...")
    clean_df = clean_data(raw_df)
    clean_df.to_csv("clean_books.csv", index=False)
    print(clean_df.head())

    print("\nBuilding SQLite database ...")
    conn = build_database(clean_df, db_path="zepto_books.db")

    print("\nRunning SQL queries ...")
    run_queries(conn)

    print("\nComparing pd.read_sql vs pd.merge ...")
    compare_read_sql_vs_merge(conn)

    conn.close()
    print("\nDone. Database saved as zepto_books.db")