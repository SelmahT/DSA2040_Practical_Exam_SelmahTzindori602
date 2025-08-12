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
