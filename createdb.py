# create_admin_table.py

import sqlite3

# Connect to your database (it will create one if it doesn't exist)
conn = sqlite3.connect('database.db')

# Create a cursor object
cursor = conn.cursor()

# Create the admin table
cursor.execute('''
CREATE TABLE IF NOT EXISTS admin (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    password TEXT NOT NULL
)
''')

# Insert default admin credentials
cursor.execute("INSERT INTO admin (username, password) VALUES (?, ?)", ('admin', 'admin123'))

# Commit the changes and close the connection
conn.commit()
conn.close()

print("✅ Admin table created and default user inserted.")

