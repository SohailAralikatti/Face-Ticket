import sqlite3

# Connect to the database (it will create it if it doesn't exist)
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Create the students table
cursor.execute('''
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    usn TEXT UNIQUE NOT NULL,
    branch TEXT NOT NULL,
    semester TEXT NOT NULL,
    hallticket TEXT UNIQUE NOT NULL
);
''')

# Commit and close
conn.commit()
conn.close()

print("✅ 'students' table created successfully.")

