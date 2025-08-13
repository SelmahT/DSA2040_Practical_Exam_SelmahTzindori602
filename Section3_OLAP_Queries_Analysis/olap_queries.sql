-- =========================================================
-- OLAP Queries for Retail Data Warehouse
-- Filename: olap_queries.sql
-- =========================================================

-- 1. ROLL-UP QUERY: Total Sales by Country and Quarter
-- Groups sales by country and quarter
SELECT 
    c.country,
    printf('%d-Q%d', strftime('%Y', t.date), ((CAST(strftime('%m', t.date) AS INTEGER)-1)/3 + 1)) AS quarter,
    SUM(f.total_sales) AS total_sales
FROM SalesFact f
JOIN CustomerDim c ON f.customer_id = c.customer_id
JOIN TimeDim t ON f.time_id = t.time_id
GROUP BY c.country, quarter
ORDER BY c.country, quarter;

-- =========================================================

-- 2. DRILL-DOWN QUERY: Monthly Sales Details for Australia
-- Provides detailed sales information per month
SELECT 
    strftime('%Y-%m', t.date) AS month,
    f.invoice_no,
    f.stock_code,
    f.description,
    f.quantity,
    f.total_sales
FROM SalesFact f
JOIN CustomerDim c ON f.customer_id = c.customer_id
JOIN TimeDim t ON f.time_id = t.time_id
WHERE c.country = 'Australia'
ORDER BY month, f.invoice_no;

-- =========================================================

-- 3. SLICE QUERY: Total Sales for Electronics Category
-- Sums sales only for products in the 'Electronics' category
SELECT 
    c.country,
    SUM(f.total_sales) AS total_sales
FROM SalesFact f
JOIN CustomerDim c ON f.customer_id = c.customer_id
JOIN TimeDim t ON f.time_id = t.time_id
WHERE f.category = 'Electronics'
GROUP BY c.country
ORDER BY total_sales DESC;

-- =========================================================
-- End of OLAP Queries
-- =========================================================
