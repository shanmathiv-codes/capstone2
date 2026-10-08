import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "data" / "telecom_ops.db"
SCHEMA_PATH = BASE_DIR / "sql" / "01_schema.sql"
SEED_PATH = BASE_DIR / "sql" / "02_seed_data.sql"

# Make sure data directory exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Connect to SQLite database
conn = sqlite3.connect(DB_PATH)

try:
    # Create tables
    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        schema_sql = file.read()

    conn.executescript(schema_sql)

    # Insert seed data
    with open(SEED_PATH, "r", encoding="utf-8") as file:
        seed_sql = file.read()

    conn.executescript(seed_sql)

    conn.commit()

    print("Database created successfully.")
    print(f"Location: {DB_PATH}")

finally:
    conn.close()