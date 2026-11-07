# Module 1: Data Pipeline (`/data_pipeline`)

## Overview
End-to-end catalog scraping, cleaning, currency enrichment, and relational storage pipeline built for Zepto's competitive-intelligence workflow.

- **Data Source**: [books.toscrape.com](https://books.toscrape.com/)
- **Categories Scraped (4 categories, 198 books total)**:
  - `Mystery` (32 books)
  - `Historical Fiction` (26 books)
  - `Sequential Art` (75 books)
  - `Fiction` (65 books)
- **Fixed Currency Conversion Baseline**: **1 GBP = 105.50 INR** (project-defined constant; keyless and deterministic).

## Setup & Execution
From the repository root:
```bash
pip install -r requirements.txt
python data_pipeline/pipeline.py
```

Running `data_pipeline/pipeline.py` executes the full workflow:
1. Scrapes all paginated pages across the 4 target categories (`198` books) from `books.toscrape.com`, saving `data_pipeline/books_raw.csv` as an offline fallback cache.
2. Cleans fields into strict types (`price_gbp: float`, `rating: int (1-5)`, `in_stock: bool`, `price_inr: float`) and saves `data_pipeline/books_cleaned.csv`.
3. Recreates `data_pipeline/books_catalog.db` with foreign-key enforcement (`PRAGMA foreign_keys = ON`) and loads the normalized `categories` and `books` tables.
4. Executes 5 analytical SQL queries covering `SELECT`, `WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN`, `IN`, and `JOIN`, and verifies that `pd.read_sql` and in-memory `pd.merge` produce identical results (`sql_join_df.equals(pandas_join_df) == True`).

## Parsing, Cleaning & Imputation Decisions
- **Non-numeric identifiers (`title`, `category`)**: Stripped of surrounding whitespace; rows missing either identifier are dropped because a catalog entry without a title or category cannot be relationally linked.
- **`price_gbp` (`float`)**: Extracted via regex `r"(\d+(?:\.\d+)?)"` after stripping currency symbols (`£`).
- **`rating` (`int 1–5`)**: Mapped from CSS word classes (`One` → `1`, `Two` → `2`, `Three` → `3`, `Four` → `4`, `Five` → `5`).
- **`in_stock` (`bool`)**: Parsed case-insensitively by checking whether the availability string contains `"in stock"`.
- **Missing / Corrupt Numeric Value Strategy**: Any numeric field (`price_gbp` or `rating`) that fails to parse is handled with **median imputation** (`df["price_gbp"].median()` and `round(df["rating"].median())`) rather than dropping the row, preserving catalog coverage while remaining robust against price outliers.
- **`price_inr` (`float`)**: Computed using the exact project baseline constant `1 GBP = 105.50 INR`, rounded to 2 decimal places.

## Relational Schema (`books_catalog.db`)
```sql
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
```

## Executed SQL Queries & Outputs

### Query 1 (`SELECT`, `WHERE`, `ORDER BY`, `LIMIT`) — Top 10 five-star in-stock books ordered by highest INR price
```sql
SELECT book_id, title, price_gbp, price_inr, rating, in_stock
FROM books
WHERE rating = 5 AND in_stock = 1
ORDER BY price_inr DESC
LIMIT 10;
```
```text
 book_id                                                                    title  price_gbp  price_inr  rating  in_stock
     180                                                         Digital Fortress      58.00    6119.00       5         1
      93                                                                 El Deafo      57.62    6078.91       5         1
     197                                            Kitchens of the Great Midwest      57.20    6034.60       5         1
     107            The Sandman, Vol. 3: Dream Country (The Sandman (volumes) #3)      55.55    5860.52       5         1
      35                                  A Flight of Arrows (The Pathfinders #2)      55.53    5858.42       5         1
     142                                 Finders Keepers (Bill Hodges Trilogy #2)      53.53    5647.42       5         1
     185                                                     The Husband's Secret      52.51    5539.80       5         1
      19 The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1)      52.30    5517.65       5         1
      59                  Scott Pilgrim's Precious Little Life (Scott Pilgrim #1)      52.29    5516.60       5         1
     141                                     The Regional Office Is Under Attack!      51.36    5418.48       5         1
```

### Query 2 (`DISTINCT`, `ORDER BY`) — Distinct star ratings and stock statuses available in the catalog
```sql
SELECT DISTINCT rating, in_stock
FROM books
ORDER BY rating DESC;
```
```text
 rating  in_stock
      5         1
      4         1
      3         1
      2         1
      1         1
```

### Query 3 (`BETWEEN`, `WHERE`, `ORDER BY`, `LIMIT`) — Books priced between 2000 and 3500 INR
```sql
SELECT book_id, title, price_gbp, price_inr, rating
FROM books
WHERE price_inr BETWEEN 2000.0 AND 3500.0
ORDER BY price_inr ASC
LIMIT 10;
```
```text
 book_id                                                   title  price_gbp  price_inr  rating
      66                             Pop Gun War, Volume 1: Gift      18.97    2001.33       1
     173                        Me Before You (Me Before You #1)      19.02    2006.61       1
      25               The Cuckoo's Calling (Cormoran Strike #1)      19.21    2026.66       1
      61                                         This One Summer      19.49    2056.19       4
      99                Fruits Basket, Vol. 7 (Fruits Basket #7)      19.57    2064.64       1
       2                                    In a Dark, Dark Wood      19.63    2070.96       1
      72 Lumberjanes Vol. 3: A Terrible Plan (Lumberjanes #9-12)      19.92    2101.56       2
      31                    Blood Defense (Samantha Brinkman #1)      20.30    2141.65       3
      40                                    Love, Lies and Spies      20.55    2168.02       2
      53                                  Between Shades of Gray      20.79    2193.34       5
```

### Query 4 (`IN`, `JOIN`, `WHERE`, `ORDER BY`, `LIMIT`) — High-rated books in Mystery or Fiction categories
```sql
SELECT b.book_id, b.title, c.category_name, b.rating, b.price_inr
FROM books b
JOIN categories c ON b.category_id = c.category_id
WHERE c.category_name IN ('Mystery', 'Fiction') AND b.rating >= 4
ORDER BY b.rating DESC, b.price_inr ASC
LIMIT 10;
```
```text
 book_id                                                             title category_name  rating  price_inr
     155                                                Dear Mr. Knightley       Fiction       5    1182.66
      29                                                 The Girl You Lost       Mystery       5    1296.59
     195                                                   The Silent Wife       Fiction       5    1301.87
     137                                                            Thirst       Fiction       5    1821.98
      23                                 The Silkworm (Cormoran Strike #2)       Mystery       5    2431.78
      18 What Happened on Beale Street (Secrets of the South Mysteries #2)       Mystery       5    2676.54
     181                                                    Atlas Shrugged       Fiction       5    2804.19
     143                                                   The Time Keeper       Fiction       5    2941.34
     178                                                        Siddhartha       Fiction       5    3610.21
     198                                                      Bright Lines       Fiction       5    4121.88
```

### Query 5 (`JOIN`, `ORDER BY`, `LIMIT`) — Top 15 highest-rated books joined with category names
```sql
SELECT b.book_id, b.title, c.category_name, b.price_gbp, b.price_inr, b.rating, b.in_stock
FROM books b
INNER JOIN categories c ON b.category_id = c.category_id
ORDER BY b.rating DESC, b.price_inr DESC, b.book_id ASC
LIMIT 15;
```

## Side-by-Side Comparison: `pd.read_sql` vs. `pd.merge`

### `pd.read_sql` Output (SQL JOIN)
```text
 book_id                                                                     title      category_name  price_gbp  price_inr  rating  in_stock
     180                                                          Digital Fortress            Fiction      58.00    6119.00       5         1
      93                                                                  El Deafo     Sequential Art      57.62    6078.91       5         1
     197                                             Kitchens of the Great Midwest            Fiction      57.20    6034.60       5         1
     107             The Sandman, Vol. 3: Dream Country (The Sandman (volumes) #3)     Sequential Art      55.55    5860.52       5         1
      35                                   A Flight of Arrows (The Pathfinders #2) Historical Fiction      55.53    5858.42       5         1
     142                                  Finders Keepers (Bill Hodges Trilogy #2)            Fiction      53.53    5647.42       5         1
     185                                                      The Husband's Secret            Fiction      52.51    5539.80       5         1
      19  The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1)            Mystery      52.30    5517.65       5         1
      59                   Scott Pilgrim's Precious Little Life (Scott Pilgrim #1)     Sequential Art      52.29    5516.60       5         1
     141                                      The Regional Office Is Under Attack!            Fiction      51.36    5418.48       5         1
     136                                              We Love You, Charlie Freeman            Fiction      50.27    5303.48       5         1
     191                                  Crazy Rich Asians (Crazy Rich Asians #1)            Fiction      49.13    5183.22       5         1
       9                                    A Time of Torment (Charlie Parker #14)            Mystery      48.35    5100.92       5         1
     135                                               Private Paris (Private #10)            Fiction      47.61    5022.85       5         1
     125 Rat Queens, Vol. 1: Sass & Sorcery (Rat Queens (Collected Editions) #1-5)     Sequential Art      46.96    4954.28       5         1
```

### `pd.merge` Output (Pure Pandas In-Memory Join)
```text
 book_id                                                                     title      category_name  price_gbp  price_inr  rating  in_stock
     180                                                          Digital Fortress            Fiction      58.00    6119.00       5         1
      93                                                                  El Deafo     Sequential Art      57.62    6078.91       5         1
     197                                             Kitchens of the Great Midwest            Fiction      57.20    6034.60       5         1
     107             The Sandman, Vol. 3: Dream Country (The Sandman (volumes) #3)     Sequential Art      55.55    5860.52       5         1
      35                                   A Flight of Arrows (The Pathfinders #2) Historical Fiction      55.53    5858.42       5         1
     142                                  Finders Keepers (Bill Hodges Trilogy #2)            Fiction      53.53    5647.42       5         1
     185                                                      The Husband's Secret            Fiction      52.51    5539.80       5         1
      19  The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1)            Mystery      52.30    5517.65       5         1
      59                   Scott Pilgrim's Precious Little Life (Scott Pilgrim #1)     Sequential Art      52.29    5516.60       5         1
     141                                      The Regional Office Is Under Attack!            Fiction      51.36    5418.48       5         1
     136                                              We Love You, Charlie Freeman            Fiction      50.27    5303.48       5         1
     191                                  Crazy Rich Asians (Crazy Rich Asians #1)            Fiction      49.13    5183.22       5         1
       9                                    A Time of Torment (Charlie Parker #14)            Mystery      48.35    5100.92       5         1
     135                                               Private Paris (Private #10)            Fiction      47.61    5022.85       5         1
     125 Rat Queens, Vol. 1: Sass & Sorcery (Rat Queens (Collected Editions) #1-5)     Sequential Art      46.96    4954.28       5         1
```

**Equivalence Verification**: `sql_join_df.equals(pandas_join_df)` evaluates to **`True`**.
