# instacart-customer-segmentation
ยินดีต้อนรับสู่ instacart-customer-segmentation!

## Project Overview
โปรเจคนี้สร้างจากชุดข้อมูล Instacart Online Grocery Basket Analysis (Kaggle) โดยมีเป้าหมายเพื่อ
- สร้างคุณสมบัติระดับผู้ใช้ (user-level features)
- วิเคราะห์และสร้างรายงานการกระจายของฟีเจอร์ (data profiling & visualization)
- แบ่งกลุ่มลูกค้าด้วยเทคนิค Clustering (K-Means)

## โครงสร้างโฟลเดอร์สำคัญ:
- data/: โฟลเดอร์เก็บข้อมูลดิบและผลลัพธ์กลาง (เช่น merged_data.csv)
- src/preprocessing/: สคริปต์เตรียมข้อมูล (รวม, ทำความสะอาด, สร้าง features)
- src/analysis/: สคริปต์สร้างรายงานและกราฟ (data profiling, visualization)
- src/modeling/: สคริปต์รัน clustering และบันทึกผล
- reports/: ผลลัพธ์ของ data profiling และรูปภาพ
- results/: ผลลัพธ์การ clustering (เช่น clustered_data.csv

## ไฟล์ที่เกี่ยวข้อง (สำคัญ)
- [src/preprocessing/datapipline.py](src/preprocessing/datapipline.py) — ดาวน์โหลดและรวมไฟล์เป็น `data/merged_data.csv`
- [src/preprocessing/preprocess.py](src/preprocessing/preprocess.py) — preprocessing, scaling, บันทึก `data/preprocessed/processed_data.csv`
- [src/preprocessing/customer_features.py](src/preprocessing/customer_features.py) — สร้างฟีเจอร์ระดับผู้ใช้ (output: `feature/user_features.csv`)
- [src/analysis/data_profile.py](src/analysis/data_profile.py) — สรุป missing/outliers และบันทึกไปที่ `reports/data_profile/`
- [src/analysis/data_visualization.py](src/analysis/data_visualization.py) — สร้างกราฟและบันทึกไปที่ `reports/data_profile/plots`
- [src/modeling/clustering.py](src/modeling/clustering.py) — หาจำนวนคลัสเตอร์ที่เหมาะสมและสร้างผลลัพธ์

## Requirements
ติดตั้ง Python 3.8+ และแพ็กเกจจาก `requirements.txt`:
```powershell
python -m venv env
env\Scripts\Activate.ps1   # PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

## ขั้นตอนการรัน (แนะนำตามลำดับ)

1) เตรียมชุดข้อมูล

- วิธีอัตโนมัติ (อาจต้องมีการล็อกอิน/ดาวน์โหลดจาก Kaggle ด้วยตนเอง):

```powershell
python src/preprocessing/datapipline.py
```

เมื่อรันคำสั่ง จะดาวน์โหลดชุดข้อมูล ตาม URL ที่ถูกกำหนดไว้ และรวมไฟล์ CSV ให้เป็น `data/merged_data.csv`.
ถ้าการดาวน์โหลดไม่สำเร็จ ให้ดาวน์โหลดไฟล์จาก Kaggle แล้ววาง CSVs ลงในโฟลเดอร์ `data/` ด้วยตนเอง แล้วรันสคริปต์อีกครั้งเพื่อสร้าง `merged_data.csv`.

2) ตรวจสอบ/รัน preprocessing

```powershell
python src/preprocessing/preprocess.py
```

คำสั่งนี้จะทำการ
- โหลด `data/merged_data.csv`
- จัดการ missing values (mean imputation)
- จัดการ outliers ด้วย IQR clipping
- ทำ Standard Scaling (ยกเว้นคอลัมน์ ID)
- บันทึก `data/preprocessed/processed_data.csv` และ `data/preprocessed/scaler.pkl`

หากต้องการสร้างเฉพาะตัวอย่าง ให้ปรับ `sample_size` ใน `preprocess.py` หรือเรียกฟังก์ชัน `preprocess_merged_data` โดยระบุพารามิเตอร์

3) สร้างฟีเจอร์ระดับผู้ใช้ (user-level features)

รันสคริปต์ต่อไปเพื่อรวบรวมฟีเจอร์ระดับผู้ใช้ (เช่น frequency, recency, reorder rate)

```powershell
python src/preprocessing/customer_features.py
```

ผลลัพธ์ที่จะได้คือ `feature/user_features.csv` (หรือไฟล์ในโฟลเดอร์ `feature/`)

4) วิเคราะห์และสร้างรายงาน Data Profile

```powershell
python src/analysis/data_profile.py
```

คำสั่งนี้จะอ่าน `data/merged_data.csv` และสร้างไฟล์ใน `reports/data_profile/` ได้แก่
- `missing_values.csv`
- `outlier_summary.csv`
- `feature_distribution_numeric.csv` และ `feature_distribution_categorical.csv`
- `feature_improvements.md`

5) สร้างกราฟ Visualization

```powershell
python src/analysis/data_visualization.py
```

ผลลัพธ์จะอยู่ที่ `reports/data_profile/plots/`

6) รัน Clustering (แบ่งกลุ่มลูกค้า)

ก่อนรัน ให้แน่ใจว่าไฟล์ฟีเจอร์ระดับผู้ใช้พร้อมและมีคอลัมน์ `user_id` (สคริปต์จะโหลด `feature/user_features.csv` เป็นค่าเริ่มต้น)

```powershell
python src/modeling/clustering.py
```

เมื่อรันคำสั่งแล้ว จะทำการ
- สเกลข้อมูล, ลดมิติด้วย PCA
- คำนวณ Elbow และ Silhouette เพื่อช่วยเลือกจำนวนคลัสเตอร์
- รัน KMeans และบันทึกผลเฉลี่ยของแต่ละคลัสเตอร์ไปที่ `results/clustered_data.csv`

## ผลลัพธ์หลัก
- `data/preprocessed/processed_data.csv` — ข้อมูลหลัง preprocessing
- `feature/user_features.csv` — ฟีเจอร์ระดับผู้ใช้ (input สำหรับ clustering)
- `reports/data_profile/` — รายงานการวิเคราะห์ข้อมูลและไฟล์ CSV สรุป
- `reports/data_profile/plots/` — รูปภาพ visualization
- `results/clustered_data.csv` — สรุปผล clustering per-cluster

## Team & Collaboration
โปรเจคนี้พัฒนาโดยทีมจำนวน 3 คน

นายปภพ สมนอก      663380018-4

นายวีรภัทร วิเศษสมบัติ  663380025-7

นายปกรณ์ จำนงค์นารถ  663380216-0	
