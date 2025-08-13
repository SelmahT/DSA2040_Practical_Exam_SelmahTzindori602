# DSA 2040 Practical Exam - Data Warehousing and Data Mining

## Section 1: Data Warehousing
### Overview

This task involved designing a data warehouse star schema for a retail company that sells products across various categories, including electronics and clothing. The company tracks sales transactions, customer details, product information, and time data.

The objective was to create a data model that supports key analytical queries such as total sales by product category per quarter, customer demographic analysis, and inventory trend tracking.

### Task 1: Data Warehouse Design

#### Star Schema Explanation

The chosen design is a **star schema** consisting of one fact table (`fact_sales`) and four dimension tables (`dim_customer`, `dim_product`, `dim_store`, and `dim_time`).

#####  Fact Table

- **fact_sales**: Captures measurable sales data including sales amount and quantity, and links to dimensions through foreign keys (`customer_id`, `product_id`, `time_id`, `store_id`).

##### Dimension Tables

- **dim_customer**: Contains customer attributes such as customer ID, name, gender, age, and location.
- **dim_product**: Stores product details like product ID, name, category, and brand.
- **dim_store**: Holds store-related information such as store ID, name, and location.
- **dim_time**: Records time information including date, quarter, and year for time-based analysis.

---

## Why Star Schema?

The star schema was selected over a snowflake schema for the following reasons:

- It simplifies querying and improves performance due to fewer joins.
- It provides a clear, intuitive structure for business users and analysts.
- It aligns well with the retail company's need for fast aggregation and reporting on sales and customer demographics.

---

## Schema Diagram
![alt text](Section1_Data_Warehousing/retail_star_schema.drawio.png)
---

### SQL Schema Script

- The SQL script file `dw_schema.sql` includes all the `CREATE TABLE` statements required to build the schema.

```bash
CREATE TABLE dim_customer (
    customer_id INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT,
    gender TEXT,
    age INTEGER
);

CREATE TABLE dim_product (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    brand TEXT
);

CREATE TABLE dim_store (
    store_id INTEGER PRIMARY KEY,
    store_name TEXT,
    location TEXT
);

CREATE TABLE dim_time (
    time_id INTEGER PRIMARY KEY,
    date TEXT,
    quarter TEXT,
    year INTEGER
);

CREATE TABLE fact_sales (
    sales_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,
    time_id INTEGER,
    store_id INTEGER,
    sales_amount REAL,
    quantity INTEGER,
    FOREIGN KEY (customer_id) REFERENCES dim_customer(customer_id),
    FOREIGN KEY (product_id) REFERENCES dim_product(product_id),
    FOREIGN KEY (time_id) REFERENCES dim_time(time_id),
    FOREIGN KEY (store_id) REFERENCES dim_store(store_id)
);

```

- To create the schema in SQLite, run:

  ```bash
  sqlite3 retail_dw.db < dw_schema.sql
  ```

- This will set up the tables and constraints needed for the data warehouse.

---

# ETL Retail Data Project

## Overview

This project performs a full **ETL (Extract, Transform, Load)** process on a synthetic retail dataset. The ETL pipeline is implemented in Python and stores the cleaned and aggregated data into a **SQLite database** (`retail_dw.db`). All steps of the process are **logged** in `etl_process.log`, including the number of rows processed at each stage.

---

## Project Contents

- **etl_retail.py** (or `etl_retail.ipynb`): Python script or notebook containing the Etl Process ETL function and logging configuration.  
- **data folder** 
this contains the extract simulated data and transformed data:

- **synthetic_retail_data.csv**: Input CSV dataset (used for extraction).

***Sample Output***
```
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
INV10629,P5569,It Of,45,2025-06-30 03:26:14,5.41,10089.0,United Kingdom
INV10788,P1423,Live Eight,35,2024-08-05 19:57:11,74.23,10062.0,Canada
INV10684,P8933,Unit Example,45,2023-12-01 22:12:30,66.75,10063.0,Australia
INV10516,P2020,Fill Relationship,1,2024-12-12 00:34:30,56.25,10034.0,Netherlands
INV10529,P9947,Though Could,24,2024-03-12 10:59:29,38.15,10023.0,Australia
INV10657,P1282,Occur Evidence,5,2025-01-19 03:15:43,49.95,10006.0,France
INV10552,P1823,What Follow,35,2025-02-14 15:01:34,3.88,10060.0,France
INV10531,P8999,Beyond Next,50,2023-09-22 01:26:54,52.89,10095.0,United Kingdom
```

- **transformed_retail_data.csv** (optional): CSV output containing transformed sales data. 

***Sample Output***

```
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country,TotalSales
INV10629,P5569,It Of,45,2025-06-30 03:26:14,5.41,10089,United Kingdom,243.45000000000002
INV10516,P2020,Fill Relationship,1,2024-12-12 00:34:30,56.25,10034,Netherlands,56.25
INV10657,P1282,Occur Evidence,5,2025-01-19 03:15:43,49.95,10006,France,249.75
INV10552,P1823,What Follow,35,2025-02-14 15:01:34,3.88,10060,France,135.79999999999998
INV10321,P5533,Push Start,14,2024-10-21 19:53:14,32.44,10065,France,454.15999999999997
INV10070,P5422,Risk Officer,18,2025-04-06 08:08:33,17.08,10033,Canada,307.43999999999994
INV10668,P7971,Executive Power,7,2024-08-25 00:55:00,72.28,10034,Netherlands,505.96000000000004
INV10247,P9883,Magazine Part,29,2025-07-16 04:39:19,38.47,10018,USA,1115.6299999999999
INV10613,P8102,Professor Forget,7,2025-02-06 04:05:45,93.8,10013,Germany,656.6

```


- **retail_dw.db**: SQLite database created by the ETL script containing the following tables:  
  - `CustomerDim` — Customer dimension table.  
  - `TimeDim` — Time dimension table.  
  - `SalesFact` — Fact table with sales records.

**Screenshots of a sample of the contents in this data base tables**
(a) Customer Dimension Table

![alt text](Section2_ETL_process/retail_tableoutputs_load/customerdimtable.PNG)

(b) Sales Fact Table

![alt text](Section2_ETL_process/retail_tableoutputs_load/sales_facttable.PNG)

(c)Time Dimension Table

![alt text](Section2_ETL_process/retail_tableoutputs_load/time_dimtable.PNG)


- **etl_process.log**: Log file containing detailed information about rows processed, transformations applied, and records inserted.

---
# ETL Process
The ETL was developed in two phases:

1. **Stepwise Execution:** The Extraction, Transformation, and Load steps were first implemented and tested separately to ensure data quality and correctness.  
2. **Function Encapsulation:** After verification, the ETL steps were combined into a single function `run_etl()` for easier execution, logging, and reusability.

---

## Step Wise Execution

---

# Step 0: Synthetic Data Generation

### Step 0: Synthetic Data Generation with Real-World Imperfections

In this step, we generate a synthetic retail dataset (~1100 rows) designed to resemble real-world data by including:

- Valid sales records with random realistic values.
- Outliers: negative or zero quantities and unit prices to simulate data entry errors.
- Duplicate records to represent accidental repeated entries.
- Missing values in some columns to mimic incomplete data.

After generation, the dataset is saved as `synthetic_retail_data.csv` to simulate an extraction source file for the ETL process.
*** Code Used ***

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import random
from faker import Faker

# Initialize Faker for realistic data generation
fake = Faker()

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define parameters
num_rows = 1000
num_customers = 100
num_countries = 7

# Generate InvoiceNo as unique invoice strings
invoice_numbers = [f"INV{10000 + i}" for i in range(num_rows)]

# Generate StockCode as product codes
stock_codes = [f"P{random.randint(1000,9999)}" for _ in range(num_rows)]

# Generate product descriptions
products = [fake.word().capitalize() + " " + fake.word().capitalize() for _ in range(num_rows)]

# Quantities: integers 1 to 50, with some negative values as outliers
quantities = np.random.randint(1, 51, size=num_rows)
outlier_indices = np.random.choice(num_rows, size=10, replace=False)
quantities[outlier_indices] = -np.random.randint(1, 20, size=10)  # negative quantities as outliers

# UnitPrice: floats between 1 and 100, with some zero or negative outliers
unit_prices = np.round(np.random.uniform(1, 100, size=num_rows), 2)
price_outlier_indices = np.random.choice(num_rows, size=5, replace=False)
unit_prices[price_outlier_indices] = np.random.uniform(-20, 0, size=5)  # negative or zero prices as outliers

# InvoiceDate: random dates over 2 years (Aug 12, 2023 to Aug 12, 2025)
start_date = pd.Timestamp('2023-08-12')
end_date = pd.Timestamp('2025-08-12')
invoice_dates = [fake.date_time_between(start_date, end_date) for _ in range(num_rows)]

# CustomerID: Use float type array to allow NaNs
customer_ids = np.random.choice(range(10000, 10000 + num_customers), size=num_rows).astype(float)

# Country: random selection from 7 countries
countries_list = ['United Kingdom', 'France', 'Germany', 'Netherlands', 'USA', 'Canada', 'Australia']
countries = np.random.choice(countries_list, size=num_rows)

# Introduce some missing CustomerID and Description values randomly
missing_customer_indices = np.random.choice(num_rows, size=20, replace=False)
for i in missing_customer_indices:
    customer_ids[i] = np.nan  # Now valid as customer_ids are floats

missing_description_indices = np.random.choice(num_rows, size=15, replace=False)
for i in missing_description_indices:
    products[i] = None

# Create DataFrame
df_synthetic = pd.DataFrame({
    'InvoiceNo': invoice_numbers,
    'StockCode': stock_codes,
    'Description': products,
    'Quantity': quantities,
    'InvoiceDate': invoice_dates,
    'UnitPrice': unit_prices,
    'CustomerID': customer_ids,
    'Country': countries
})

# Add some duplicate rows by duplicating random samples
duplicates = df_synthetic.sample(10, random_state=42)
df_synthetic = pd.concat([df_synthetic, duplicates], ignore_index=True)

# Shuffle the dataset
df_synthetic = df_synthetic.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the generated dataset to CSV for extraction stage
df_synthetic.to_csv('synthetic_retail_data.csv', index=False)

print("Synthetic dataset generated and saved as 'synthetic_retail_data.csv'.")
print(f"Dataset shape: {df_synthetic.shape}")
print(f"Number of duplicates: {df_synthetic.duplicated().sum()}")
print(f"Number of missing CustomerID: {df_synthetic['CustomerID'].isna().sum()}")
print(f"Number of missing Description: {df_synthetic['Description'].isna().sum()}")

```

---


# Set Up the Logger

```python
import pandas as pd
import logging
import sys

# ------------------ Logger Setup ------------------
logger = logging.getLogger("ETLLogger")
logger.setLevel(logging.INFO)

# Prevent multiple handlers in notebooks
if not logger.handlers:
    # File handler
    file_handler = logging.FileHandler("etl_process.log", mode='a')  # append logs
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info("Logger initialized successfully")

```

- Our logger was initilized successfully

---

# Step 1: Extract

- In this step, we read the synthetic CSV data generated previously into a pandas DataFrame. 
- We will handle missing values and data types (convert InvoiceDate to datetime).
- This prepares the raw data for transformation.

*** Code Used ***

```python
logger.info("ETL Process started - Extraction step")

# Load synthetic CSV dataset
df = pd.read_csv('synthetic_retail_data.csv')

# Convert InvoiceDate to datetime format for time-based operations
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Drop rows where critical columns have missing values
missing_before = df.shape[0]
df = df.dropna(subset=['CustomerID', 'Description'])
missing_after = df.shape[0]
logger.info(f"Dropped {missing_before - missing_after} rows due to missing CustomerID or Description")

# Remove duplicate rows
duplicates_before = df.shape[0]
df = df.drop_duplicates()
duplicates_after = df.shape[0]
logger.info(f"Removed {duplicates_before - duplicates_after} duplicate rows")

# Ensure proper data types
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0).astype(int)
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce').fillna(0.0).astype(float)
df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce').astype('Int64')

logger.info(f"Extracted data shape after cleaning: {df.shape}")
logger.info(f"Missing values after cleaning:\n{df.isna().sum()}")

# Optional: show first 5 rows
print(df.head())

```

*** Our output ***
Output Analysis:

-  The dataset was successfully loaded from the CSV file.
-  InvoiceDate was converted to datetime format for easier filtering later.
-  The dataset contains missing CustomerID and Description values, as expected from the synthetic generation.
-  There are also duplicate rows, which will be handled during transformation.
## Extraction Step Outputs

| Timestamp                  | Log Message |
|-----------------------------|------------|
| 2025-08-13 09:29:30,395    | ETL Process started - Extraction step |
| 2025-08-13 09:29:30,419    | Dropped 35 rows due to missing CustomerID or Description |
| 2025-08-13 09:29:30,428    | Removed 10 duplicate rows |
| 2025-08-13 09:29:30,435    | Extracted data shape after cleaning: (965, 8) |
| 2025-08-13 09:29:30,444    | Missing values after cleaning: <br>InvoiceNo: 0, StockCode: 0, Description: 0, Quantity: 0, InvoiceDate: 0, UnitPrice: 0, CustomerID: 0, Country: 0 |

### Sample Extracted Data (First 5 Rows)

| InvoiceNo | StockCode | Description        | Quantity | InvoiceDate          | UnitPrice | CustomerID | Country        |
|-----------|-----------|------------------|----------|--------------------|-----------|------------|----------------|
| INV10629  | P5569     | It Of            | 45       | 2025-06-30 03:26:14 | 5.41      | 10089      | United Kingdom |
| INV10788  | P1423     | Live Eight       | 35       | 2024-08-05 19:57:11 | 74.23     | 10062      | Canada         |
| INV10684  | P8933     | Unit Example     | 45       | 2023-12-01 22:12:30 | 66.75     | 10063      | Australia      |
| INV10516  | P2020     | Fill Relationship | 1       | 2024-12-12 00:34:30 | 56.25     | 10034      | Netherlands   |
| INV10529  | P9947     | Though Could     | 24       | 2024-03-12 10:59:29 | 38.15     | 10023      | Australia      |

---

## Step 2.1: Save Transformed Data to CSV
- After cleaning and transforming the data (calculating TotalSales, filtering, etc.),
- we save the resulting DataFrame to a CSV file named 'transformed_retail_data.csv'. as seen above
- This allows for easy data sharing and serves as an intermediate checkpoint.

---

# Step 3: Load (Inserting Data into SQLite Database)

In this step, we will load the transformed retail data into a SQLite database named `retail_dw.db`.

We will create three tables:
- `CustomerDim` to store unique customers,
- `TimeDim` to store unique dates and time attributes,
- `SalesFact` to store the sales transactions linked to the customer and time dimensions via foreign keys.

This design follows the data warehousing star schema pattern and satisfies the project requirement to load data into a fact table and at least two dimension tables.

*** Code Used ***

```python
import sqlite3
import pandas as pd
import logging

# Logger assumed already configured at the start of your script

logger.info("Load step started")

# Connect or create SQLite DB
conn = sqlite3.connect('retail_dw.db')
cursor = conn.cursor()

# Create dimension and fact tables if not exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS CustomerDim (
    customer_id INTEGER PRIMARY KEY,
    country TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS TimeDim (
    time_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT UNIQUE,
    year INTEGER,
    month INTEGER,
    day INTEGER
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS SalesFact (
    sales_id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_no TEXT,
    stock_code TEXT,
    description TEXT,
    quantity INTEGER,
    unit_price REAL,
    total_sales REAL,
    customer_id INTEGER,
    time_id INTEGER,
    FOREIGN KEY(customer_id) REFERENCES CustomerDim(customer_id),
    FOREIGN KEY(time_id) REFERENCES TimeDim(time_id)
)
''')

conn.commit()
logger.info("Tables created or verified")

# Insert unique customers into CustomerDim
customers = df[['CustomerID', 'Country']].drop_duplicates().dropna()
customers.columns = ['customer_id', 'country']

for _, row in customers.iterrows():
    cursor.execute('''
        INSERT OR IGNORE INTO CustomerDim (customer_id, country) VALUES (?, ?)
    ''', (int(row['customer_id']), row['country']))
conn.commit()
logger.info(f"Inserted {customers.shape[0]} customers into CustomerDim")

# Insert unique dates into TimeDim
dates = df[['InvoiceDate']].drop_duplicates()
dates['year'] = dates['InvoiceDate'].dt.year
dates['month'] = dates['InvoiceDate'].dt.month
dates['day'] = dates['InvoiceDate'].dt.day
dates['date_str'] = dates['InvoiceDate'].dt.strftime('%Y-%m-%d')

for _, row in dates.iterrows():
    cursor.execute('''
        INSERT OR IGNORE INTO TimeDim (date, year, month, day) VALUES (?, ?, ?, ?)
    ''', (row['date_str'], row['year'], row['month'], row['day']))
conn.commit()
logger.info(f"Inserted {dates.shape[0]} dates into TimeDim")

# Get time_id mapping safely
time_map_df = pd.read_sql_query("SELECT time_id, date FROM TimeDim", conn)
time_map_df.rename(columns={'date':'date_str'}, inplace=True)

# Prepare merge keys
df['date_str'] = df['InvoiceDate'].dt.strftime('%Y-%m-%d')

# Drop any existing 'time_id' columns to avoid suffix issues
if 'time_id' in df.columns:
    df.drop(columns=['time_id'], inplace=True)

# Merge df with time_map_df on 'date_str'
df = df.merge(time_map_df, how='left', on='date_str')

# Check for missing time_id after merge
missing_time_ids = df['time_id'].isna().sum()
if missing_time_ids > 0:
    logger.warning(f"{missing_time_ids} records have missing time_id after merge and will be skipped.")

# Insert sales records, skipping rows with missing foreign keys
sales_inserted = 0
for _, row in df.iterrows():
    if pd.isna(row['CustomerID']) or pd.isna(row['time_id']):
        continue  # Skip rows with missing keys
    
    cursor.execute('''
        INSERT INTO SalesFact (
            invoice_no, stock_code, description, quantity, unit_price, total_sales, customer_id, time_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        row['InvoiceNo'],
        row['StockCode'],
        row['Description'],
        int(row['Quantity']),
        float(row['UnitPrice']),
        float(row['TotalSales']),
        int(row['CustomerID']),
        int(row['time_id'])
    ))
    sales_inserted += 1

conn.commit()
logger.info(f"Inserted {sales_inserted} sales records into SalesFact")

# Optional: Print counts to console
for table in ['CustomerDim', 'TimeDim', 'SalesFact']:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"Table {table} has {count} records.")

conn.close()
logger.info("Load step completed")

```

*** Output analysis an explanation ***

## Load Step Outputs

| Timestamp                  | Log Message |
|-----------------------------|------------|
| 2025-08-13 09:30:42,365    | Load step started |
| 2025-08-13 09:30:42,374    | Tables created or verified |
| 2025-08-13 09:30:42,440    | Inserted 343 customers into CustomerDim |
| 2025-08-13 09:30:42,653    | Inserted 476 dates into TimeDim |
| 2025-08-13 09:30:42,900    | Inserted 476 sales records into SalesFact |
| 2025-08-13 09:30:42,909    | Load step completed |

### Table Counts After Load

| Table Name     | Record Count |
|----------------|--------------|
| CustomerDim    | 100          |
| TimeDim        | 351          |
| SalesFact      | 3302         |

---

## ETL Function Usage

The ETL logic is encapsulated in the function:
# Full ETL Function: Extraction, Transformation, and Load (ETL)

This function `run_etl()` performs the complete ETL process on the synthetic retail dataset, including:

1. **Extraction**:
   - Loads the CSV dataset into a Pandas DataFrame.
   - Converts `InvoiceDate` to datetime format.
   - Drops rows with missing critical values (`CustomerID`, `Description`).
   - Removes duplicate rows.
   - Ensures proper data types for numerical columns (`Quantity`, `UnitPrice`, `CustomerID`).
   - Logs the number of rows processed and missing values.

2. **Transformation**:
   - Calculates a new column `TotalSales` as `Quantity * UnitPrice`.
   - Removes outliers: rows with negative `Quantity` or non-positive `UnitPrice`.
   - Filters data to include sales within the last year (Aug 13, 2024 – Aug 12, 2025).
   - Creates a **customer summary table** with total purchases per customer and country information.
   - Logs the number of rows processed at each step.

3. **Load**:
   - Connects to the SQLite database (`retail_dw.db`) and creates tables if they do not exist:
     - `CustomerDim` (customer dimension table)
     - `TimeDim` (time dimension table)
     - `SalesFact` (sales fact table)
   - Inserts unique customers into `CustomerDim`.
   - Inserts unique invoice dates into `TimeDim`.
   - Maps `InvoiceDate` to `time_id` for linking fact and dimension tables.
   - Inserts processed sales records into `SalesFact`.
   - Logs any missing `time_id` values and the number of rows inserted at each table.
   - Closes the database connection.

**Logging**:
- All ETL steps are logged to both the console and a log file `etl_process.log`.
- Logs include timestamps, log levels, and messages about rows processed, outliers removed, and warnings if any.

This function ensures that the ETL pipeline is fully traceable, robust against missing or invalid data, and produces clean, ready-to-use data in the warehouse.
*** Code Used***
```python
import pandas as pd
import sqlite3
import logging

# =========================
# Configure logging
# =========================
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler for log file
file_handler = logging.FileHandler('etl_process.log', mode='w')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler for immediate output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(file_formatter)
logger.addHandler(console_handler)

# =========================
# Define ETL function
# =========================
def run_etl(csv_file='synthetic_retail_data.csv', db_file='retail_dw.db'):
    # --- Extraction ---
    logger.info("ETL Process started - Extraction step")
    
    # Load CSV into DataFrame
    df = pd.read_csv(csv_file)
    
    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    
    # Drop rows missing critical info
    missing_before = df.shape[0]
    df = df.dropna(subset=['CustomerID', 'Description'])
    missing_after = df.shape[0]
    logger.info(f"Dropped {missing_before - missing_after} rows due to missing CustomerID or Description")
    
    # Remove duplicate rows
    duplicates_before = df.shape[0]
    df = df.drop_duplicates()
    duplicates_after = df.shape[0]
    logger.info(f"Removed {duplicates_before - duplicates_after} duplicate rows")
    
    # Ensure proper numeric types
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0).astype(int)
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce').fillna(0.0).astype(float)
    df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce').astype('Int64')  # nullable int
    
    logger.info(f"Extracted data shape after cleaning: {df.shape}")
    logger.info(f"Missing values after cleaning:\n{df.isna().sum()}")
    
    # --- Transformation ---
    logger.info("Transformation step started")
    
    # Calculate TotalSales
    df['TotalSales'] = df['Quantity'] * df['UnitPrice']
    logger.info("Calculated TotalSales column")
    
    # Remove outliers (Quantity < 0 or UnitPrice <= 0)
    initial_shape = df.shape
    df = df[(df['Quantity'] >= 0) & (df['UnitPrice'] > 0)]
    logger.info(f"Removed outliers: {initial_shape[0] - df.shape[0]} rows dropped")
    
    # Filter for last year sales
    start_date = pd.to_datetime('2024-08-13')
    end_date = pd.to_datetime('2025-08-12')
    df = df[(df['InvoiceDate'] >= start_date) & (df['InvoiceDate'] <= end_date)]
    logger.info(f"Filtered data for sales in last year: shape now {df.shape}")
    
    # Create customer summary (dimension-like)
    customer_summary = df.groupby('CustomerID').agg(
        TotalPurchases=pd.NamedAgg(column='TotalSales', aggfunc='sum'),
        Country=pd.NamedAgg(column='Country', aggfunc=lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
    ).reset_index()
    logger.info(f"Created customer summary with {customer_summary.shape[0]} unique customers")
    
    # --- Load ---
    logger.info("Load step started")
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create dimension and fact tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS CustomerDim (
            customer_id INTEGER PRIMARY KEY,
            country TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS TimeDim (
            time_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            year INTEGER,
            month INTEGER,
            day INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS SalesFact (
            sales_id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_no TEXT,
            stock_code TEXT,
            description TEXT,
            quantity INTEGER,
            unit_price REAL,
            total_sales REAL,
            customer_id INTEGER,
            time_id INTEGER,
            FOREIGN KEY(customer_id) REFERENCES CustomerDim(customer_id),
            FOREIGN KEY(time_id) REFERENCES TimeDim(time_id)
        )
    ''')
    conn.commit()
    logger.info("Tables created or verified")
    
    # Insert unique customers
    customers = customer_summary[['CustomerID', 'Country']].rename(columns={'CustomerID': 'customer_id'})
    for _, row in customers.iterrows():
        cursor.execute('INSERT OR IGNORE INTO CustomerDim (customer_id, country) VALUES (?, ?)',
                       (int(row['customer_id']), row['Country']))
    conn.commit()
    logger.info(f"Inserted {customers.shape[0]} customers into CustomerDim")
    
    # Insert unique dates into TimeDim
    dates = df[['InvoiceDate']].drop_duplicates()
    dates['year'] = dates['InvoiceDate'].dt.year
    dates['month'] = dates['InvoiceDate'].dt.month
    dates['day'] = dates['InvoiceDate'].dt.day
    dates['date_str'] = dates['InvoiceDate'].dt.strftime('%Y-%m-%d')
    
    for _, row in dates.iterrows():
        cursor.execute('INSERT OR IGNORE INTO TimeDim (date, year, month, day) VALUES (?, ?, ?, ?)',
                       (row['date_str'], row['year'], row['month'], row['day']))
    conn.commit()
    logger.info(f"Inserted {dates.shape[0]} dates into TimeDim")
    
    # Map InvoiceDate to time_id
    time_map_df = pd.read_sql_query("SELECT time_id, date FROM TimeDim", conn)
    time_map_df.rename(columns={'date':'date_str'}, inplace=True)
    df['date_str'] = df['InvoiceDate'].dt.strftime('%Y-%m-%d')
    df = df.merge(time_map_df, how='left', on='date_str')
    
    if df['time_id'].isnull().any():
        missing_time_ids = df[df['time_id'].isnull()].shape[0]
        logger.warning(f"{missing_time_ids} sales records have missing time_id after merge")
    
    # Insert sales into SalesFact
    sales_inserted = 0
    for _, row in df.iterrows():
        if pd.isna(row['CustomerID']) or pd.isna(row['time_id']):
            continue
        cursor.execute('''
            INSERT INTO SalesFact (
                invoice_no, stock_code, description, quantity, unit_price, total_sales, customer_id, time_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (row['InvoiceNo'], row['StockCode'], row['Description'], int(row['Quantity']),
              float(row['UnitPrice']), float(row['TotalSales']), int(row['CustomerID']), int(row['time_id'])))
        sales_inserted += 1
    conn.commit()
    logger.info(f"Inserted {sales_inserted} sales records into SalesFact")
    
    # Optional: log counts in each table
    for table in ['CustomerDim', 'TimeDim', 'SalesFact']:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        logger.info(f"Table {table} has {count} records")
    
    conn.close()
    logger.info("ETL Process completed successfully")
    return df, customer_summary
```

- Run this code to call the function

```python
df, customer_summary = run_etl()
