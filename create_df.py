import pandas as pd
import sqlite3
import os
import logging
# from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# load_dotenv()

conn = sqlite3.connect('debt_recovery.db')

# Querying and joining tables
query = '''
SELECT 
    d.debt_id, 
    dt.name, dt.credit_score,
    d.original_amount, d.current_amount, d.date_incurred, d.due_date, d.status,
    p.amount_paid, p.payment_date,
    c.date AS last_communication_date, c.method AS last_communication_method, c.response_received,
    l.action_type, l.date_initiated, l.status AS legal_action_status
FROM 
    Debts d
    JOIN Debtors dt ON d.debtor_id = dt.debtor_id
    LEFT JOIN Payments p ON d.debt_id = p.debt_id
    LEFT JOIN CommunicationLog c ON dt.debtor_id = c.debtor_id
    LEFT JOIN LegalActions l ON d.debt_id = l.debt_id;
'''

df = pd.read_sql_query(query, conn)
# Close the database connection
conn.close()

# Display the DataFrame
print(df)

def clean_data(dataframe):
    try:
        dataframe.fillna(method='ffill', inplace=True)
        dataframe.drop_duplicates(subset=['case_id'], inplace=True)
        dataframe['last_payment_date'] = pd.to_datetime(dataframe['last_payment_date'])
        dataframe = dataframe[dataframe['debt_amount'] >= 0]
        return dataframe
    except Exception as e:
        logging.error(f"Data cleaning error: {e}")
        raise

# df = clean_data(df)
    
# Fill missing values or drop them based on your requirement
df.fillna(method='ffill', inplace=True)  # Forward fill
# df.dropna(inplace=True)  # Drop rows with missing values

# Convert date columns to datetime objects
date_columns = ['date_incurred', 'due_date', 'payment_date', 'last_communication_date', 'date_initiated']
for col in date_columns:
    df[col] = pd.to_datetime(df[col])

# Example: Calculate debt age
df['debt_age'] = (pd.Timestamp.now() - df['date_incurred']).dt.days

# Additional feature engineering based on domain knowledge

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Identify numerical and categorical columns
numerical_cols = ['credit_score', 'original_amount', 'current_amount', 'amount_paid', 'debt_age']
categorical_cols = ['status', 'last_communication_method', 'action_type', 'legal_action_status']

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Apply transformations
df_transformed = preprocessor.fit_transform(df)

# Creating the transformed DataFrame
transformed_df = pd.DataFrame(df_transformed, columns=numerical_cols + categorical_cols)

# Now you can save this DataFrame to CSV
transformed_df.to_csv('file1.csv', index=False)
