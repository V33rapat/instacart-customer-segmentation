# Data Dictionary

## Customer Behavioral Features
เอกสารนี้อธิบายความหมายของแต่ละ Feature ที่ใช้ในการวิเคราะห์และทำ Clustering ผู้ใช้งาน

This document describes each feature used for customer analysis and clustering.

| Feature               | Type  | Description (English)                             | คำอธิบาย (ภาษาไทย)                              |
| --------------------- | ----- | ------------------------------------------------- | ----------------------------------------------- |
| recency               | int   | Number of days since the customer's last order    | จำนวนวันตั้งแต่ลูกค้าทำการสั่งซื้อครั้งล่าสุด   |
| frequency_orders      | int   | Total number of orders made by the customer       | จำนวนคำสั่งซื้อทั้งหมดของลูกค้า                 |
| frequency_items       | int   | Total number of items purchased                   | จำนวนสินค้าทั้งหมดที่ลูกค้าซื้อ                 |
| monetary_total        | int | Total monetary value spent by the customer        | ยอดเงินรวมทั้งหมดที่ลูกค้าใช้จ่าย               |
| monetary_avg          | float | Average spending per order                        | ค่าเฉลี่ยยอดใช้จ่ายต่อคำสั่งซื้อ                |
| monetary_std          | int | Standard deviation of spending per order          | ส่วนเบี่ยงเบนมาตรฐานของยอดใช้จ่ายต่อคำสั่งซื้อ  |
| num_departments       | int   | Number of unique departments purchased from       | จำนวนหมวดหมู่หลัก (Department) ที่ลูกค้าเคยซื้อ |
| num_aisles            | int   | Number of unique aisles purchased from            | จำนวนหมวดหมู่ย่อย (Aisle) ที่ลูกค้าเคยซื้อ      |
| num_unique_products   | int   | Number of unique products purchased               | จำนวนสินค้าที่ไม่ซ้ำกันที่ลูกค้าเคยซื้อ         |
| order_dow_min         | int   | Minimum day of week customer placed an order      | วันที่ในสัปดาห์ที่ลูกค้าเคยสั่งซื้อเร็วที่สุด   |
| order_dow_max         | int   | Maximum day of week customer placed an order      | วันที่ในสัปดาห์ที่ลูกค้าเคยสั่งซื้อล่าสุด       |
| order_dow_mean        | float | Average day of week customer places orders        | ค่าเฉลี่ยวันในสัปดาห์ที่ลูกค้าสั่งซื้อ          |
| order_hour_min        | int   | Earliest hour of the day customer placed an order | ชั่วโมงที่ลูกค้าเคยสั่งซื้อเร็วที่สุด           |
| order_hour_max        | int   | Latest hour of the day customer placed an order   | ชั่วโมงที่ลูกค้าเคยสั่งซื้อล่าสุด               |
| order_hour_mean       | float | Average hour of the day customer places orders    | ค่าเฉลี่ยชั่วโมงที่ลูกค้าสั่งซื้อ               |
| days_since_prior_mean | float | Average number of days between orders             | ค่าเฉลี่ยจำนวนวันระหว่างคำสั่งซื้อแต่ละครั้ง    |
| days_since_prior_min  | float   | Minimum days between orders                       | จำนวนวันต่ำสุดระหว่างคำสั่งซื้อ                 |
| days_since_prior_max  | float   | Maximum days between orders                       | จำนวนวันสูงสุดระหว่างคำสั่งซื้อ                 |
