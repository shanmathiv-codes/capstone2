import sqlite3

conn = sqlite3.connect("data/telecom_ops.db")

cursor = conn.cursor()

cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name;
""")

tables = cursor.fetchall()

print("Tables:")
for table in tables:
    print("-", table[0])

conn.close()