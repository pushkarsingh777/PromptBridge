import sqlite3

conn = sqlite3.connect("data/memory.db")

# Show all tables
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("=== Tables ===")
print(tables)

# Show schema of each table
for (table,) in tables:
    if table == "sqlite_sequence":
        continue
    print(f"\n=== Schema: {table} ===")
    schema = conn.execute(f"PRAGMA table_info({table})").fetchall()
    for col in schema:
        print(" ", col)
    print(f"\n=== Data: {table} (last 20) ===")
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 20").fetchall()
    for row in rows:
        print(row)

conn.close()
