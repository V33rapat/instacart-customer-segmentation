import pandas as pd
import os
import urllib.request

# ตั้งค่าให้แสดงทุกคอลัมน์
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# URLs ของไฟล์ CSV (สามารถเพิ่มลิงค์ได้)
urls = [
    {
        "url": "https://www.kaggle.com/api/v1/datasets/download/yasserh/instacart-online-grocery-basket-analysis-dataset",
        "filename": "dataset.zip"
    }
    # เพิ่มลิงค์อื่นๆ ที่นี่:
    # {
    #     "url": "https://example.com/data.zip",
    #     "filename": "data.zip"
    # }
]

download_path = "./data"

# สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(download_path, exist_ok=True)

# ดาวน์โหลด (ถ้ายังไม่มีไฟล์)
for url_item in urls:
    url = url_item["url"]
    filename = url_item["filename"]
    filepath = os.path.join(download_path, filename)
    
    if not os.path.exists(filepath):
        print(f"กำลังดาวน์โหลด {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            
            # แตกไฟล์ zip ถ้าเป็นไฟล์ zip
            if filename.endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(download_path)
            
            print(f"ดาวน์โหลด {filename} สำเร็จ!")
        except Exception as e:
            print(f"ไม่สามารถดาวน์โหลด {filename} ได้: {e}")
            print("ให้โหลดไฟล์ลงมาแล้ววางในโฟลเดอร์ ./data ด้วยตนเอง")
    else:
        print(f"{filename} มีอยู่แล้ว")

# อ่านไฟล์ CSV ทั้งหมดและเก็บเป็น dictionary
csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
dataframes = {}

if csv_files:
    print("กำลังอ่านไฟล์...")
    for csv_file in csv_files:
        filepath = os.path.join(download_path, csv_file)
        df = pd.read_csv(filepath)
        dataframes[csv_file] = df
        
        print(f"\n{'='*50}")
        print(f"ไฟล์: {csv_file}")
        print(f"{'='*50}")
        print(f"ขนาดข้อมูล: {df.shape}")
        print(f"คอลัมน์: {df.columns.tolist()}")
        print(f"ชนิดข้อมูล:\n{df.dtypes}")
    
    # ================== JOIN ตารางตามโครงสร้าง ==================
    print(f"\n\n{'='*70}")
    print("เริ่มการ JOIN ตารางตามโครงสร้าง")
    print(f"{'='*70}\n")
    
    # หาไฟล์ order_products (อาจชื่อต่างกัน)
    order_products_file = None
    for key in dataframes.keys():
        if 'order_products' in key.lower():
            order_products_file = key
            break
    
    if order_products_file:
        print(f"พบไฟล์ order_products: {order_products_file}")
        
        # 1. JOIN order_products กับ products
        if 'products.csv' in dataframes:
            print("\n1. JOIN order_products กับ products...")
            merged_df = dataframes[order_products_file].merge(
                dataframes['products.csv'],
                on='product_id',
                how='left'
            )
            print(f"   ผลลัพธ์: {merged_df.shape}")
            
            # 2. JOIN ผลลัพธ์กับ aisles
            if 'aisles.csv' in dataframes:
                print("\n2. JOIN ผลลัพธ์กับ aisles...")
                merged_df = merged_df.merge(
                    dataframes['aisles.csv'],
                    on='aisle_id',
                    how='left'
                )
                print(f"   ผลลัพธ์: {merged_df.shape}")
            
            # 3. JOIN ผลลัพธ์กับ departments
            if 'departments.csv' in dataframes:
                print("\n3. JOIN ผลลัพธ์กับ departments...")
                merged_df = merged_df.merge(
                    dataframes['departments.csv'],
                    on='department_id',
                    how='left'
                )
                print(f"   ผลลัพธ์: {merged_df.shape}")
            
            # 4. JOIN ผลลัพธ์กับ orders
            if 'orders.csv' in dataframes:
                print("\n4. JOIN ผลลัพธ์กับ orders...")
                merged_df = merged_df.merge(
                    dataframes['orders.csv'],
                    on='order_id',
                    how='left'
                )
                print(f"   ผลลัพธ์: {merged_df.shape}")
            
            # บันทึกผลลัพธ์
            output_file = os.path.join(download_path, "merged_data.csv")
            merged_df.to_csv(output_file, index=False)
            print(f"\n✅ บันทึกข้อมูลรวมเรียบร้อย: {output_file}")
            print(f"ขนาด: {merged_df.shape}")
            print(f"\nตัวอย่าง 5 แถวแรก:")
            print(merged_df.head())
        else:
            print("❌ ไม่พบไฟล์ products.csv")
    else:
        print("❌ ไม่พบไฟล์ order_products")
else:
    print("ไม่พบไฟล์ CSV โปรดโหลดลงมาด้วยตนเอง")