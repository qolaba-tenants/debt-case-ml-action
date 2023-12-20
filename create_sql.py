import sqlite3
import pandas as pd

# Establishing a connection to the SQLite database
conn = sqlite3.connect('debt_recovery.db')

# Function to execute each SQL command
def execute_sql_command(command):
    try:
        conn.execute(command)
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

# Creating tables one by one
tables_commands = [
    '''CREATE TABLE IF NOT EXISTS Debtors (
        debtor_id INTEGER PRIMARY KEY,
        name TEXT,
        address TEXT,
        phone TEXT,
        email TEXT,
        credit_score INTEGER
    );''',

    '''CREATE TABLE IF NOT EXISTS Debts (
        debt_id INTEGER PRIMARY KEY,
        debtor_id INTEGER,
        original_amount REAL,
        current_amount REAL,
        date_incurred TEXT,
        due_date TEXT,
        status TEXT,
        FOREIGN KEY (debtor_id) REFERENCES Debtors(debtor_id)
    );''',

    '''CREATE TABLE IF NOT EXISTS Payments (
        payment_id INTEGER PRIMARY KEY,
        debt_id INTEGER,
        amount_paid REAL,
        payment_date TEXT,
        FOREIGN KEY (debt_id) REFERENCES Debts(debt_id)
    );''',

    '''CREATE TABLE IF NOT EXISTS CommunicationLog (
        log_id INTEGER PRIMARY KEY,
        debtor_id INTEGER,
        date TEXT,
        method TEXT,
        response_received BOOLEAN,
        FOREIGN KEY (debtor_id) REFERENCES Debtors(debtor_id)
    );''',

    '''CREATE TABLE IF NOT EXISTS LegalActions (
        action_id INTEGER PRIMARY KEY,
        debt_id INTEGER,
        action_type TEXT,
        date_initiated TEXT,
        status TEXT,
        FOREIGN KEY (debt_id) REFERENCES Debts(debt_id)
    );'''
]

for command in tables_commands:
    execute_sql_command(command)

# Inserting sample data
conn.execute("INSERT INTO Debtors (name, address, phone, email, credit_score) VALUES ('John Doe', '123 Main St', '555-1234', 'johndoe@example.com', 680)")
conn.execute("INSERT INTO Debts (debtor_id, original_amount, current_amount, date_incurred, due_date, status) VALUES (1, 5000, 4500, '2021-01-01', '2021-12-31', 'not_recovered')")
conn.execute("INSERT INTO Payments (debt_id, amount_paid, payment_date) VALUES (1, 500, '2021-06-01')")
conn.execute("INSERT INTO CommunicationLog (debtor_id, date, method, response_received) VALUES (1, '2021-05-20', 'email', 1)")
conn.execute("INSERT INTO LegalActions (debt_id, action_type, date_initiated, status) VALUES (1, 'notice', '2021-08-01', 'pending')")
conn.commit()
# Close the database connection
conn.close()

