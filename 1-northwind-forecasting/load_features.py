# load_features.py

import pandas as pd
from sqlalchemy import create_engine, text

# PostgreSQL bağlantı URI'sı — bilgilerinizi girin
db_uri = "postgresql://postgres:fatma@localhost:5432/GYK1Northwind"
engine = create_engine(db_uri)

def load_customer_snapshots():
    query = """
    WITH params AS (
        SELECT INTERVAL '6 months' AS horizon
    ),
    order_totals AS (
        SELECT
            o.order_id,
            o.customer_id,
            o.order_date,
            SUM(od.unit_price * od.quantity * (1 - od.discount)) AS order_value
        FROM orders o
        JOIN order_details od USING (order_id)
        GROUP BY o.order_id, o.customer_id, o.order_date
    ),
    snapshots AS (
        SELECT
            ot.*,
            SUM(order_value) OVER (PARTITION BY customer_id ORDER BY order_date)  AS total_spent,
            ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date)      AS total_orders,
            AVG(order_value) OVER (PARTITION BY customer_id ORDER BY order_date)  AS avg_order_value,
            (ot.order_date
                - LAG(ot.order_date) OVER (PARTITION BY customer_id ORDER BY order_date)
            )::INT                                                               AS days_since_prev_order,
            EXTRACT(MONTH FROM order_date)                                       AS order_month
        FROM order_totals ot
    )
    SELECT
        s.customer_id,
        s.order_id           AS snapshot_id,
        s.order_date         AS snapshot_date,
        COALESCE(s.total_spent, 0)            AS total_spent,
        COALESCE(s.total_orders, 0)           AS total_orders,
        COALESCE(s.avg_order_value, 0)        AS avg_order_value,
        COALESCE(s.days_since_prev_order, 0)  AS days_since_prev_order,
        s.order_month,
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM order_totals fut, params p
                WHERE fut.customer_id = s.customer_id
                  AND fut.order_date  > s.order_date
                  AND fut.order_date <= s.order_date + p.horizon
            ) THEN 1
            ELSE 0
        END AS reorder_6m
    FROM snapshots s
    ORDER BY s.customer_id, s.order_date;
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn)
    return df

if __name__ == "__main__":
    df_snap = load_customer_snapshots()
    print(df_snap.head())
    print(f"Toplam satır: {len(df_snap)}")
