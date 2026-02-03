import sqlite3

# Connect to your database
conn = sqlite3.connect("database.db")  # or "/mnt/data/database.db" if running in notebook
cursor = conn.cursor()

# SQL commands to create the new tables
sql_commands = [
    """
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        usn TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS fraud (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
]

# Execute each command
for command in sql_commands:
    cursor.execute(command)

# Commit and close
conn.commit()
conn.close()

print("✅ Tables created successfully.")

