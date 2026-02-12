"""
Simple preprocessing script - จัดการ Missing Values, Outliers, Scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def preprocess_merged_data(input_path='data/merged_data.csv', 
                          output_path='data/preprocessed/processed_data.csv',
                          scaler_path='data/preprocessed/scaler.pkl',
                          sample_size=None):
    """
    ประมวลผลข้อมูลเต็มรูป:
    1. โหลดข้อมูล
    2. จัดการ Missing Values
    3. ตรวจหาและจัดการ Outliers
    4. Standard Scaling
    5. บันทึกผลลัพธ์
    """
    
    print("="*70)
    print("📊 DATA PREPROCESSING")
    print("="*70)
    
    # 1. โหลดข้อมูล
    print("\n[1/5] โหลดข้อมูล...")
    if sample_size:
        df = pd.read_csv(input_path, nrows=sample_size)
        print(f"✅ โหลด {sample_size} rows: {df.shape}")
    else:
        df = pd.read_csv(input_path)
        print(f"✅ โหลดสำเร็จ: {df.shape}")
    
    original_shape = df.shape
    
    # 2. จัดการ Missing Values
    print("\n[2/5] จัดการ Missing Values...")
    missing_before = df.isnull().sum()
    missing_cols = missing_before[missing_before > 0]
    
    if len(missing_cols) > 0:
        print(f"   พบ missing ใน {len(missing_cols)} คอลัมน์:")
        for col, count in missing_cols.items():
            pct = count / len(df) * 100
            print(f"   - {col}: {count} ({pct:.2f}%)")
        
        # เติม missing ด้วย mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='mean')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        print(f"   ✅ เติมด้วย mean strategy")
    else:
        print(f"   ✅ ไม่มี missing values")
    
    # 3. จัดการ Outliers (IQR method)
    print("\n[3/5] จัดการ Outliers (IQR method)...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_count = 0
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                outlier_count += outliers
                df[col] = df[col].clip(lower=lower, upper=upper)
    
    print(f"   ✅ จัดการ {outlier_count} outliers (clipped)")
    
    # 4. Standard Scaling (ยกเว้น ID columns)
    print("\n[4/5] ทำ Standard Scaling...")
    # คอลัมน์ ID ที่ไม่ต้อง scale
    id_cols = ['user_id', 'order_id', 'product_id']
    cols_to_scale = [col for col in numeric_cols if col not in id_cols]
    
    if len(cols_to_scale) > 0:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        print(f"   ✅ Scaled {len(cols_to_scale)} feature columns")
        print(f"   ⏭️  ทำเว้น ID columns: {', '.join([c for c in id_cols if c in df.columns])}")
    else:
        scaler = StandardScaler()
        print(f"   ⏭️  ไม่มีคอลัมน์ที่ต้อง scale")
    
    # 5. บันทึกผลลัพธ์
    print("\n[5/5] บันทึกผลลัพธ์...")
    
    # สร้าง output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # บันทึก processed data
    df.to_csv(output_path, index=False)
    print(f"   ✅ บันทึก: {output_path}")
    print(f"      Shape: {df.shape}")
    
    # บันทึก scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✅ บันทึก scaler: {scaler_path}")
    
    # บันทึกรายงาน
    report_path = output_dir / 'preprocessing_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("DATA PREPROCESSING REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Original shape: {original_shape}\n")
        f.write(f"Final shape: {df.shape}\n\n")
        f.write(f"Preprocessing steps:\n")
        f.write(f"1. Missing Values: {len(missing_cols)} columns handled\n")
        f.write(f"2. Outliers: {outlier_count} values clipped (IQR method)\n")
        f.write(f"3. Scaling: StandardScaler applied to feature columns (excluded: user_id, order_id, product_id)\\n\\n")
        f.write(f"Output files:\n")
        f.write(f"- {output_path}\n")
        f.write(f"- {scaler_path}\n")
    
    print(f"   ✅ บันทึกรายงาน: {report_path}")
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\n📊 สรุป:")
    print(f"   Original: {original_shape}")
    print(f"   Processed: {df.shape}")
    print(f"   Missing values handled: {len(missing_cols)}")
    print(f"   Outliers clipped: {outlier_count}")
    print(f"\n💾 Output files:")
    print(f"   - {output_path}")
    print(f"   - {scaler_path}")
    print(f"   - {report_path}\n")
    
    return df, scaler


if __name__ == '__main__':
    # รัน preprocessing
    df_processed, scaler = preprocess_merged_data(
        input_path='data/merged_data.csv',
        output_path='data/preprocessed/processed_data.csv',
        scaler_path='data/preprocessed/scaler.pkl',
        sample_size=None  # ใช้ทั้งหมด (หรือใส่ตัวเลขสำหรับ sample)
    )
    
    print("📌 ตัวอย่างข้อมูล (5 rows แรก):")
    print(df_processed.head())
    print(f"\n📌 Data types:")
    print(df_processed.dtypes)
