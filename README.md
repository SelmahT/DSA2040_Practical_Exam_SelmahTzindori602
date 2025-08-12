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
Section1_Data_Warehousing/retail_star_schema.drawio.png

---

### SQL Schema Script

- The SQL script file `dw_schema.sql` includes all the `CREATE TABLE` statements required to build the schema.
- To create the schema in SQLite, run:
  ```bash
  sqlite3 retail_dw.db < dw_schema.sql
  ```

- This will set up the tables and constraints needed for the data warehouse.

---