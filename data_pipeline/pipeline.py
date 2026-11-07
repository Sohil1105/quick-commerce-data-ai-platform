import sqlite3
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR_RATE = 105.50
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
OUT_DIR = Path(__file__).resolve().parent
DB_PATH = OUT_DIR / "books_catalog.db"
RAW_CSV_PATH = OUT_DIR / "books_raw.csv"
CSV_PATH = OUT_DIR / "books_cleaned.csv"
QUERIES_OUT_PATH = OUT_DIR / "query_results.txt"

TARGET_CATEGORIES = [
    ("Mystery", "catalogue/category/books/mystery_3/index.html"),
    ("Historical Fiction", "catalogue/category/books/historical-fiction_4/index.html"),
    ("Sequential Art", "catalogue/category/books/sequential-art_5/index.html"),
    ("Fiction", "catalogue/category/books/fiction_10/index.html"),
]


def scrape_category(session: requests.Session, category_name: str, rel_url: str) -> list[dict]:
    records = []
    next_url = urljoin(BASE_URL, rel_url)
    while next_url:
        resp = session.get(next_url, timeout=20)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        for article in soup.select("article.product_pod"):
            title_el = article.select_one("h3 a")
            price_el = article.select_one("p.price_color")
            rating_el = article.select_one("p.star-rating")
            avail_el = article.select_one("p.instock.availability")

            title = title_el.get("title", "").strip() if title_el else None
            price_raw = price_el.get_text(strip=True) if price_el else None
            star_classes = [c for c in (rating_el.get("class") or []) if c != "star-rating"] if rating_el else []
            star_rating = star_classes[0] if star_classes else None
            availability = avail_el.get_text(strip=True) if avail_el else None

            records.append(
                {
                    "title": title,
                    "price": price_raw,
                    "star_rating": star_rating,
                    "availability": availability,
                    "category": category_name,
                }
            )
        next_link = soup.select_one("li.next a")
        next_url = urljoin(next_url, next_link["href"]) if next_link and next_link.get("href") else None
    return records


def scrape_all_categories() -> pd.DataFrame:
    rows = []
    try:
        with requests.Session() as session:
            for cat_name, rel_url in TARGET_CATEGORIES:
                cat_rows = scrape_category(session, cat_name, rel_url)
                print(f"Scraped {len(cat_rows)} books from category '{cat_name}'")
                rows.extend(cat_rows)
        raw_df = pd.DataFrame(rows)
        raw_df.to_csv(RAW_CSV_PATH, index=False, encoding="utf-8")
        return raw_df
    except Exception as exc:
        print(f"[Offline Fallback] Live scraping network unavailable ({exc}); loading committed raw dataset...")
        if RAW_CSV_PATH.exists():
            return pd.read_csv(RAW_CSV_PATH, encoding="utf-8")
        if CSV_PATH.exists():
            cleaned = pd.read_csv(CSV_PATH, encoding="utf-8")
            return cleaned[["title", "price", "star_rating", "availability", "category"]]
        raise


def clean_and_convert(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # Drop rows missing essential non-numeric identifiers (title or category)
    df["title"] = df["title"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[(df["title"] != "") & (df["category"] != "") & df["title"].notna() & df["category"].notna()].copy()

    # 1. Strip currency symbol from price and convert to float price_gbp
    df["price_gbp"] = df["price"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)

    # 2. Convert text star rating (One..Five) into integer rating (1-5)
    df["rating"] = df["star_rating"].map(RATING_MAP)

    # 3. Parse availability text into boolean in_stock
    df["in_stock"] = df["availability"].astype(str).str.lower().str.contains("in stock", na=False)

    # 4. Handle failed numeric parses with median imputation
    if df["price_gbp"].isna().any():
        df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())
    if df["rating"].isna().any():
        df["rating"] = df["rating"].fillna(round(df["rating"].median()))

    df["price_gbp"] = df["price_gbp"].astype(float)
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    # 5. Convert price_gbp to price_inr using fixed project constant (1 GBP = 105.50 INR)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR_RATE).round(2)

    return df.reset_index(drop=True)


def load_to_sqlite(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(
            """
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL,
                in_stock INTEGER NOT NULL,
                category_id INTEGER NOT NULL REFERENCES categories(category_id)
            );
            """
        )

        unique_categories = sorted(df["category"].unique())
        conn.executemany(
            "INSERT INTO categories (category_name) VALUES (?);",
            [(c,) for c in unique_categories],
        )

        cat_map = dict(conn.execute("SELECT category_name, category_id FROM categories;").fetchall())
        book_tuples = (
            df.assign(category_id=df["category"].map(cat_map), in_stock=df["in_stock"].astype(int))[
                ["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]
            ]
            .itertuples(index=False, name=None)
        )
        conn.executemany(
            """
            INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            book_tuples,
        )
        conn.commit()


SQL_QUERIES = {
    "Query 1 (SELECT, WHERE, ORDER BY, LIMIT) - Top 10 five-star in-stock books ordered by highest INR price": """
        SELECT book_id, title, price_gbp, price_inr, rating, in_stock
        FROM books
        WHERE rating = 5 AND in_stock = 1
        ORDER BY price_inr DESC
        LIMIT 10;
    """,
    "Query 2 (DISTINCT, ORDER BY) - Distinct star ratings and stock statuses available in the catalog": """
        SELECT DISTINCT rating, in_stock
        FROM books
        ORDER BY rating DESC;
    """,
    "Query 3 (BETWEEN, WHERE, ORDER BY, LIMIT) - Books priced between 2000 and 3500 INR": """
        SELECT book_id, title, price_gbp, price_inr, rating
        FROM books
        WHERE price_inr BETWEEN 2000.0 AND 3500.0
        ORDER BY price_inr ASC
        LIMIT 10;
    """,
    "Query 4 (IN, JOIN, WHERE, ORDER BY, LIMIT) - High-rated books in Mystery or Fiction categories": """
        SELECT b.book_id, b.title, c.category_name, b.rating, b.price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE c.category_name IN ('Mystery', 'Fiction') AND b.rating >= 4
        ORDER BY b.rating DESC, b.price_inr ASC
        LIMIT 10;
    """,
    "Query 5 (JOIN, ORDER BY, LIMIT) - Top 15 highest-rated books joined with category names": """
        SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating, b.in_stock
        FROM books b
        INNER JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.price_inr DESC, b.book_id ASC
        LIMIT 15;
    """,
}


def run_queries_and_verify(db_path: Path = DB_PATH) -> str:
    log_lines = []
    with sqlite3.connect(db_path) as conn:
        for title, sql in SQL_QUERIES.items():
            clean_sql = "\n".join(line.strip() for line in sql.strip().splitlines())
            res_df = pd.read_sql(clean_sql, conn)
            block = (
                f"=== {title} ===\n"
                f"SQL:\n{clean_sql}\n\n"
                f"Output ({len(res_df)} rows):\n{res_df.to_string(index=False)}\n"
            )
            log_lines.append(block)

        # Read back full tables into in-memory DataFrames to compare SQL JOIN vs pd.merge()
        books_mem = pd.read_sql("SELECT * FROM books;", conn)
        categories_mem = pd.read_sql("SELECT * FROM categories;", conn)
        sql_join_df = pd.read_sql(
            SQL_QUERIES["Query 5 (JOIN, ORDER BY, LIMIT) - Top 15 highest-rated books joined with category names"],
            conn,
        )

    # Replicate Query 5 in pure Pandas via pd.merge() (no SQL)
    pandas_join_df = (
        pd.merge(books_mem, categories_mem, on="category_id", how="inner")[
            ["book_id", "title", "category_name", "price_gbp", "price_inr", "rating", "in_stock"]
        ]
        .sort_values(by=["rating", "price_inr", "book_id"], ascending=[False, False, True])
        .head(15)
        .reset_index(drop=True)
    )

    matches_exactly = sql_join_df.equals(pandas_join_df)
    comparison_block = (
        "=== SQL JOIN (pd.read_sql) vs. In-Memory Pandas Merge (pd.merge) Comparison ===\n"
        f"-- pd.read_sql Output --\n{sql_join_df.to_string(index=False)}\n\n"
        f"-- pd.merge Output --\n{pandas_join_df.to_string(index=False)}\n\n"
        f"Equivalence Check (sql_join_df.equals(pandas_join_df)): {matches_exactly}\n"
    )
    log_lines.append(comparison_block)

    assert matches_exactly, "Mismatch between SQL JOIN and pd.merge outputs!"
    full_report = "\n".join(log_lines)
    QUERIES_OUT_PATH.write_text(full_report, encoding="utf-8")
    return full_report


def main() -> None:
    print("Step 1: Scraping books from books.toscrape.com...")
    raw_df = scrape_all_categories()
    print(f"Total raw rows scraped: {len(raw_df)} across {raw_df['category'].nunique()} categories")

    print("Step 2: Cleaning and converting currency (1 GBP = 105.50 INR)...")
    clean_df = clean_and_convert(raw_df)
    clean_df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    # Self-checks for acceptance criteria
    assert len(clean_df) >= 60, f"Expected >= 60 books, got {len(clean_df)}"
    assert clean_df["category"].nunique() >= 3, "Expected >= 3 categories"
    assert pd.api.types.is_float_dtype(clean_df["price_gbp"])
    assert pd.api.types.is_integer_dtype(clean_df["rating"])
    assert pd.api.types.is_bool_dtype(clean_df["in_stock"])
    assert pd.api.types.is_float_dtype(clean_df["price_inr"])

    print("Step 3: Loading normalized data into SQLite database...")
    load_to_sqlite(clean_df, DB_PATH)

    print("Step 4: Executing SQL queries and verifying pd.read_sql vs pd.merge equivalence...")
    report = run_queries_and_verify(DB_PATH)
    print(report)
    print(f"Pipeline complete! Saved DB to {DB_PATH} and query outputs to {QUERIES_OUT_PATH}")


if __name__ == "__main__":
    main()
