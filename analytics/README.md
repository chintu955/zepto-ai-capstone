# Data Pipeline Module — Design Notes

## Fixed Conversion Rate
`1 GBP = 105.50 INR` — a project-defined constant (no live/historical lookup),
applied directly in `clean_data()` to produce `price_inr` from `price_gbp`.

## Cleaning Decisions
- Rows with an unparseable `title` or `availability` field are **dropped**, since
  these are identifying/categorical fields that cannot be reasonably imputed.
- Rows with an unparseable `price_gbp` or `rating` (numeric fields) are
  **median-imputed** rather than dropped, to preserve as much of the scraped
  catalogue as possible — a missing price or rating doesn't invalidate the rest
  of that row's data the way a missing title would.

## Schema
- `categories(category_id PK, category_name UNIQUE)`
- `books(book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK → categories.category_id)`

## SQL Queries
All 5 required queries (SELECT/WHERE, ORDER BY + LIMIT, DISTINCT, BETWEEN + IN,
and a JOIN across both tables) are defined in `run_queries()` inside
`scrape_and_load.py`. Running the script automatically prints each query's SQL
text and output to the console **and** writes the same to `query_log.txt` in
this folder, so the executed output is preserved in the repository as required
(see acceptance criteria: "≥ 5 SQL queries are present with their printed/logged
output").

## pd.read_sql vs pd.merge
`compare_read_sql_vs_merge()` reproduces the JOIN query's result two ways —
once via `pd.read_sql(...)` directly against SQLite, and once via `pd.merge(...)`
on the two tables loaded fully into memory as DataFrames — and prints both side
by side to confirm they match.

## How to Regenerate Everything From Scratch
```bash
cd data_pipeline
pip install -r requirements.txt
python scrape_and_load.py
```
This produces (all included in this folder): `raw_books.csv`, `clean_books.csv`,
`zepto_books.db`, and `query_log.txt`.