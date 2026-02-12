import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import os
warnings.filterwarnings('ignore')


class CustomerSegmentation:
    """Class สำหรับสร้าง features และแบ่งกลุ่มลูกค้า"""
    
    def __init__(self, merged_df):
        """
        Args:
            merged_df: DataFrame ที่ได้จากการ JOIN ข้อมูลทั้งหมด
        """
        self.df = merged_df.copy()
        self.customer_features = None
        self.clusters = None
    
    def create_rfm_features(self):
        """
        สร้าง RFM (Recency, Frequency, Monetary) features
        - Recency: จำนวนวันที่ผ่านไปตั้งแต่ซื้อครั้งสุดท้าย
        - Frequency: จำนวนครั้งที่ลูกค้าซื้อสินค้า
        - Monetary: ยอดเงินที่ลูกค้าใช้ทั้งหมด
        """
        print("กำลังสร้าง RFM Features...")
        
        # ดึง order_date จากคอลัมน์ที่มี (merged_data ไม่มี order_date)
        # ใช้ order_number แทนการนับลำดับการซื้อ
        # หา max order_number ให้เป็น reference
        reference_order = self.df['order_number'].max()
        
        # สร้าง RFM
        rfm = self.df.groupby('user_id').agg({
            'order_number': lambda x: reference_order - x.max(),  # Recency (ล่าสุดเท่าไหร่)
            'order_id': 'nunique',  # Frequency (จำนวน orders)
            'product_id': 'count'  # จำนวน products
        }).rename(columns={
            'order_number': 'recency',
            'order_id': 'frequency_orders',
            'product_id': 'frequency_items'
        })
        
        # Monetary: จำนวนสินค้าทั้งหมดต่อ user
        # (คำนวณจาก product_id count เนื่องจาก merged_data ไม่มี price column)
        monetary = self.df.groupby('user_id').agg({
            'product_id': 'count',  # จำนวนสินค้าทั้งหมด
            'order_id': 'nunique'    # จำนวนครั้งที่สั่งซื้อ
        }).rename(columns={
            'product_id': 'monetary_total',
            'order_id': 'num_orders'
        })
        
        # คำนวณจำนวนสินค้าเฉลี่ยต่อ order
        monetary['monetary_avg'] = monetary['monetary_total'] / monetary['num_orders']
        
        # คำนวณ standard deviation ของจำนวนสินค้าต่อ order
        def calc_items_std(group_df):
            items_per_order = group_df.groupby('order_id')['product_id'].count()
            return items_per_order.std() if len(items_per_order) > 1 else 0.0
        
        monetary['monetary_std'] = self.df.groupby('user_id').apply(calc_items_std)
        monetary = monetary.drop('num_orders', axis=1)  # ลบ temporary column
        
        # รวม RFM
        rfm = rfm.join(monetary)
        
        print(f"✅ RFM Features สำเร็จ: {rfm.shape}")
        return rfm
    
    def create_product_features(self):
        """
        สร้าง features เกี่ยวกับสินค้า:
        - จำนวน department ที่ซื้อ
        - จำนวน aisle ที่ซื้อ
        - จำนวนสินค้าที่ซื้อ
        """
        print("กำลังสร้าง Product Features...")
        
        product_features = self.df.groupby('user_id').agg({
            'department_id': 'nunique',  # จำนวน departments
            'aisle_id': 'nunique',  # จำนวน aisles
            'product_id': 'nunique'  # จำนวน products ที่ต่างกัน
        }).rename(columns={
            'department_id': 'num_departments',
            'aisle_id': 'num_aisles',
            'product_id': 'num_unique_products'
        })
        
        product_features = product_features.fillna(0)
        
        print(f"✅ Product Features สำเร็จ: {product_features.shape}")
        return product_features
    
    def create_order_pattern_features(self):
        """
        สร้าง features เกี่ยวกับรูปแบบการซื้อ:
        - เวลาที่ซื้อ (วัน, ชั่วโมง)
        - อัตราการซื้อซ้ำ
        """
        print("กำลังสร้าง Order Pattern Features...")
        
        # สร้าง features แบบ simple aggregation
        pattern_features = pd.DataFrame()
        
        # Order dow features
        dow_features = self.df.groupby('user_id')['order_dow'].agg(['min', 'max', 'mean']).rename(
            columns={'min': 'order_dow_min', 'max': 'order_dow_max', 'mean': 'order_dow_mean'})
        pattern_features = pattern_features.join(dow_features) if len(pattern_features) > 0 else dow_features
        
        # Order hour features
        hour_features = self.df.groupby('user_id')['order_hour_of_day'].agg(['min', 'max', 'mean']).rename(
            columns={'min': 'order_hour_min', 'max': 'order_hour_max', 'mean': 'order_hour_mean'})
        pattern_features = pattern_features.join(hour_features)
        
        # Days since prior order features
        days_features = self.df.groupby('user_id')['days_since_prior_order'].agg(['mean', 'min', 'max']).rename(
            columns={'mean': 'days_since_prior_mean', 'min': 'days_since_prior_min', 'max': 'days_since_prior_max'})
        pattern_features = pattern_features.join(days_features)
        
        pattern_features = pattern_features.fillna(0)
        
        print(f"✅ Order Pattern Features สำเร็จ: {pattern_features.shape}")
        return pattern_features
    
    def create_all_features(self):
        """
        สร้าง features ทั้งหมด
        """
        print("="*60)
        print("สร้าง Customer Features สำหรับ Segmentation")
        print("="*60)
        
        # สร้าง features แต่ละประเภท
        rfm = self.create_rfm_features()
        product_features = self.create_product_features()
        pattern_features = self.create_order_pattern_features()
        
        # รวม features ทั้งหมด
        self.customer_features = rfm.join([product_features, pattern_features])
        
        print(f"\n✅ รวม Features ทั้งหมด: {self.customer_features.shape}")
        print(f"\nคอลัมน์ที่สร้าง:")
        print(self.customer_features.columns.tolist())
        
        return self.customer_features
    
    def normalize_features(self, features=None):
        """
        ทำการ normalize features โดยใช้ StandardScaler
        
        Args:
            features: list ของ column ที่ต้องการ normalize
        """
        if self.customer_features is None:
            print("❌ กรุณาสร้าง features ก่อน (create_all_features)")
            return None
        
        print("กำลังทำการ Normalize Features...")
        
        df_normalized = self.customer_features.copy()
        
        # ใช้ features ทั้งหมด ถ้าไม่ระบุ
        if features is None:
            features = df_normalized.columns
        
        # Normalize
        scaler = StandardScaler()
        df_normalized[features] = scaler.fit_transform(df_normalized[features])
        
        print(f"✅ Normalize สำเร็จ")
        return df_normalized, scaler
    
    def segment_customers(self, n_clusters=4, normalized_df=None):
        """
        แบ่งกลุ่มลูกค้าโดยใช้ K-Means Clustering
        
        Args:
            n_clusters: จำนวนกลุ่ม
            normalized_df: DataFrame ที่ได้จากการ normalize
        """
        if normalized_df is None:
            normalized_df, _ = self.normalize_features()
        
        print(f"\nกำลังแบ่งกลุ่มลูกค้าเป็น {n_clusters} กลุ่ม...")
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(normalized_df)
        
        # เพิ่ม cluster ลงใน customer features
        self.customer_features['cluster'] = self.clusters
        
        print(f"✅ Clustering สำเร็จ")
        print(f"\nการกระจายของกลุ่ม:")
        print(self.customer_features['cluster'].value_counts().sort_index())
        
        return self.customer_features
    
    def analyze_segments(self):
        """
        วิเคราะห์คุณลักษณะของแต่ละกลุ่ม
        """
        if self.customer_features is None or 'cluster' not in self.customer_features.columns:
            print("❌ กรุณาทำ segment_customers ก่อน")
            return None
        
        print("\n" + "="*60)
        print("วิเคราะห์คุณลักษณะของแต่ละกลุ่ม")
        print("="*60)
        
        # เลือกเฉพาะ features ที่มีอยู่
        cols_to_analyze = [col for col in ['recency', 'frequency_orders', 'monetary_total', 
                                            'num_departments', 'num_unique_products', 
                                            'order_hour_mean', 'days_since_prior_mean'] 
                           if col in self.customer_features.columns]
        
        if len(cols_to_analyze) > 0:
            analysis = self.customer_features.groupby('cluster')[cols_to_analyze].agg(['mean', 'median']).round(2)
            print(analysis)
        else:
            print("❌ ไม่พบ features สำหรับวิเคราะห์")
        
        return None
    
    def save_features(self, output_path='feature/user_features.csv'):
        """
        บันทึก customer features ลงไฟล์
        """
        if self.customer_features is None:
            print("❌ ไม่มี features ที่สร้าง")
            return
        
        # สร้างโฟลเดอร์ถ้าไม่มี
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.customer_features.to_csv(output_path)
        print(f"✅ บันทึก features สำเร็จ: {output_path}")


# ===================== ฟังก์ชั่นช่วยเหลือ =====================

def load_merged_data(filepath='data/preprocessed/processed_data.csv'):
    """โหลด preprocessed data จากไฟล์"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ โหลดข้อมูล preprocessed สำเร็จ: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ ไม่พบไฟล์: {filepath}")
        return None


def create_customer_segments(merged_df, n_clusters=4):
    """
    ฟังก์ชั่นหลักสำหรับสร้าง features และแบ่งกลุ่มลูกค้า
    
    Args:
        merged_df: DataFrame ที่ได้จากการ JOIN
        n_clusters: จำนวนกลุ่ม
    
    Returns:
        customer_features: DataFrame ที่มี features และ cluster
    """
    # สร้าง segmentation object
    segmentation = CustomerSegmentation(merged_df)
    
    # สร้าง features
    segmentation.create_all_features()
    
    # Normalize features
    normalized_df, scaler = segmentation.normalize_features()
    
    # Segment customers
    customer_features = segmentation.segment_customers(
        n_clusters=n_clusters,
        normalized_df=normalized_df
    )
    
    # วิเคราะห์กลุ่ม
    segmentation.analyze_segments()
    
    # บันทึกผลลัพธ์ (อัตโนมัติบันทึกเป็น feature/user_features.csv)
    segmentation.save_features()
    
    return customer_features, segmentation


# ===================== Main Execution =====================

if __name__ == "__main__":
    # โหลดข้อมูล
    merged_data = load_merged_data()
    
    if merged_data is not None:
        # สร้าง customer segments
        features, segmentation = create_customer_segments(merged_data, n_clusters=4)
        
        print("\n" + "="*60)
        print("สรุปผลลัพธ์")
        print("="*60)
        print(features.head(10))
    else:
        print("❌ ไม่สามารถโหลดข้อมูลได้")
