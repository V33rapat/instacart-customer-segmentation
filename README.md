# instacart-customer-segmentation
ยินดีต้อนรับสู่ instacart-customer-segmentation!

## Project Overview
โปรเจคนี้มีวัตถุประสงค์เพื่อ วิเคราะห์และแบ่งกลุ่มลูกค้า (Customer Segmentation) จากพฤติกรรมการสั่งซื้อสินค้าออนไลน์ โดยใช้ข้อมูลจาก Instacart Online Grocery Basket Analysis Dataset จาก Kaggle โดยทีมผู้พัฒนาได้นำข้อมูลการสั่งซื้อย้อนหลังของลูกค้า มาสร้าง Feature ระดับผู้ใช้ (User-level Features) และใช้เทคนิค Unsupervised Machine Learning (K-Means Clustering) เพื่อค้นหารูปแบบพฤติกรรมการซื้อที่แตกต่างกันในกลุ่มลูกค้า

## ผลลัพธ์ของโปรเจคคือ:
-  กลุ่มลูกค้าที่มีลักษณะพฤติกรรมชัดเจน
-  การอธิบายลักษณะของแต่ละกลุ่มในเชิงธุรกิจ
-  Insight ที่สามารถนำไปใช้ต่อยอดด้านการตลาดหรือกลยุทธ์ทางธุรกิจ

## Project Objectives
-  วิเคราะห์พฤติกรรมการสั่งซื้อของลูกค้า Instacart
-  สร้างชุดข้อมูลเชิงวิเคราะห์ในระดับลูกค้า (User-level Dataset)
-  แบ่งกลุ่มลูกค้าด้วยเทคนิค Clustering
-  อธิบายความแตกต่างของแต่ละกลุ่มในเชิงพฤติกรรมและธุรกิจ

## Project Structure
- data/                       raw data and merged_data.csv (ignored)
- src/analysis/data_profile.py data profiling script (feature distributions, missing values, outliers)
- reports/                    generated outputs (summary CSVs + markdown)

## Dataset Description
โปรเจคนี้ใช้ข้อมูลจาก Instacart Online Grocery Basket Analysis Dataset
ซึ่งเป็นชุดข้อมูลพฤติกรรมการสั่งซื้อสินค้าออนไลน์ของผู้ใช้งานจริง

## ไฟล์ข้อมูลที่ใช้
orders.csv                  ข้อมูลคำสั่งซื้อของผู้ใช้ เช่น เวลา วัน และลำดับการสั่ง

order_products__prior.csv   รายการสินค้าที่ผู้ใช้เคยสั่งซื้อในอดีต

products.csv                รายละเอียดสินค้า

aisles.csv                  หมวดหมู่ย่อยของสินค้า

departments.csv	            หมวดหมู่หลักของสินค้า

## Data Profiling Workflow
1. ใช้ datapipline.py เพื่อดาวน์โหลดและรวมข้อมูลเป็น data/merged_data.csv
2. รันสคริปต์โปรไฟล์ข้อมูลเพื่อดูการกระจาย feature, missing values, และ outliers:
   python src/analysis/data_profile.py
3. ผลลัพธ์จะอยู่ที่ reports/data_profile/

## Team & Collaboration
โปรเจคนี้พัฒนาโดยทีมจำนวน 3 คน

นายปภพ สมนอก      663380018-4

นายวีรภัทร วิเศษสมบัติ  663380025-7

นายปกรณ์ จำนงค์นารถ  663380216-0	
