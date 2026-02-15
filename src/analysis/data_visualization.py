"""
สคริปต์สำหรับสร้างกราฟแสดงการกระจายของข้อมูลและวิเคราะห์ความสัมพันธ์ระหว่าง features
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# นำเข้า DatasetPaths จาก data_profile เพื่อใช้ pattern เดียวกัน
from src.analysis.data_profile import DatasetPaths


@dataclass(frozen=True)
class PlotConfig:
    """การตั้งค่าสำหรับการสร้างกราฟ"""

    # ขนาดและคุณภาพของภาพ
    dpi: int = 150
    figsize_numeric: tuple[int, int] = (16, 20)
    figsize_categorical: tuple[int, int] = (12, 10)
    figsize_correlation: tuple[int, int] = (10, 8)

    # จำนวน bins สำหรับ histogram
    histogram_bins: int = 50

    # สี palette
    palette: str = "husl"
    correlation_cmap: str = "RdBu_r"

    # Features ที่มีความหมายเชิงพฤติกรรม (ไม่รวม ID columns)
    behavioral_features: tuple[str, ...] = (
        "add_to_cart_order",
        "reordered",
        "order_number",
        "order_dow",
        "order_hour_of_day",
        "days_since_prior_order",
    )

    # ชื่อคอลัมน์ department
    department_column: str = "department"


def load_merged_data(paths: DatasetPaths) -> pd.DataFrame:
    """
    โหลดข้อมูล merged_data.csv

    Args:
        paths: DatasetPaths object ที่มี path ไปยังไฟล์ข้อมูล

    Returns:
        DataFrame ที่โหลดจากไฟล์
    """
    if not paths.merged_path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ข้อมูลที่ {paths.merged_path}. กรุณารัน datapipline.py ก่อน"
        )

    print(f"กำลังโหลดข้อมูลจาก {paths.merged_path}...")
    df = pd.read_csv(paths.merged_path)
    print(f"โหลดข้อมูลสำเร็จ: {df.shape[0]:,} แถว, {df.shape[1]} คอลัมน์")
    return df


def plot_numeric_distributions(
    df: pd.DataFrame,
    config: PlotConfig,
    output_dir: Path,
) -> Path:
    """
    สร้างกราฟแสดงการกระจายของ numeric features
    แต่ละ feature จะมี histogram และ box plot

    Args:
        df: DataFrame ข้อมูล
        config: PlotConfig การตั้งค่า
        output_dir: โฟลเดอร์สำหรับบันทึกภาพ

    Returns:
        Path ไปยังไฟล์ภาพที่บันทึก
    """
    print("\n" + "=" * 60)
    print("สร้างกราฟการกระจายของ Numeric Features")
    print("=" * 60)

    # เลือกเฉพาะ behavioral features ที่มีในข้อมูล
    features = [f for f in config.behavioral_features if f in df.columns]
    n_features = len(features)

    if n_features == 0:
        print("⚠️ ไม่พบ behavioral features ในข้อมูล")
        return output_dir / "numeric_distributions.png"

    print(f"พบ {n_features} features: {features}")

    # สร้าง subplot grid: 2 แถวต่อ feature (histogram + boxplot)
    fig, axes = plt.subplots(
        nrows=n_features,
        ncols=2,
        figsize=config.figsize_numeric,
    )

    # ตั้งค่า style
    sns.set_style("whitegrid")
    colors = sns.color_palette(config.palette, n_features)

    for idx, feature in enumerate(features):
        feature_values = df[feature].dropna()
        color = colors[idx]

        # Histogram (คอลัมน์ซ้าย)
        ax_hist = axes[idx, 0] if n_features > 1 else axes[0]
        sns.histplot(
            data=feature_values.to_numpy(),
            bins=config.histogram_bins,
            kde=True,
            color=color,
            ax=ax_hist,
        )
        ax_hist.set_title(f"Distribution of {feature}", fontsize=12, fontweight="bold")
        ax_hist.set_xlabel(feature)
        ax_hist.set_ylabel("Count")

        # เพิ่ม statistics text
        stats_text = (
            f"Mean: {feature_values.mean():.2f}\n"
            f"Std: {feature_values.std():.2f}\n"
            f"Median: {feature_values.median():.2f}"
        )
        ax_hist.text(
            0.95,
            0.95,
            stats_text,
            transform=ax_hist.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Box plot (คอลัมน์ขวา)
        ax_box = axes[idx, 1] if n_features > 1 else axes[1]
        sns.boxplot(
            x=feature_values.to_numpy(),
            color=color,
            ax=ax_box,
        )
        ax_box.set_title(f"Box Plot of {feature}", fontsize=12, fontweight="bold")
        ax_box.set_xlabel(feature)

        print(
            f"  ✓ {feature}: mean={feature_values.mean():.2f}, std={feature_values.std():.2f}, "
            f"min={feature_values.min():.0f}, max={feature_values.max():.0f}"
        )

    plt.tight_layout()

    # บันทึกภาพ
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "numeric_distributions.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close()

    print(f"\n✅ บันทึกกราฟ numeric distributions: {output_path}")
    return output_path


def plot_categorical_distributions(
    df: pd.DataFrame,
    config: PlotConfig,
    output_dir: Path,
) -> Path:
    """
    สร้างกราฟแสดงการกระจายของ categorical features (department)
    ใช้ horizontal bar chart เรียงตามจำนวน

    Args:
        df: DataFrame ข้อมูล
        config: PlotConfig การตั้งค่า
        output_dir: โฟลเดอร์สำหรับบันทึกภาพ

    Returns:
        Path ไปยังไฟล์ภาพที่บันทึก
    """
    print("\n" + "=" * 60)
    print("สร้างกราฟการกระจายของ Categorical Features (Department)")
    print("=" * 60)

    if config.department_column not in df.columns:
        print(f"⚠️ ไม่พบคอลัมน์ {config.department_column} ในข้อมูล")
        return output_dir / "categorical_distributions.png"

    # นับจำนวนแต่ละ department และเรียงลำดับ
    dept_counts = df[config.department_column].value_counts()
    print(f"พบ {len(dept_counts)} departments")

    # สร้างกราฟ
    fig, ax = plt.subplots(figsize=config.figsize_categorical)

    # สร้าง color palette ตามจำนวน
    colors = sns.color_palette(config.palette, len(dept_counts))

    # Horizontal bar chart
    dept_values: np.ndarray = dept_counts.values  # type: ignore[assignment]
    bars = ax.barh(
        y=dept_counts.index,
        width=dept_values,
        color=colors,
    )

    # เพิ่ม labels แสดงจำนวนและเปอร์เซ็นต์
    total = dept_counts.sum()
    for bar, count in zip(bars, dept_values):
        pct = (count / total) * 100
        ax.text(
            bar.get_width() + total * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.1f}%)",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Number of Orders", fontsize=12)
    ax.set_ylabel("Department", fontsize=12)
    ax.set_title(
        "Distribution of Orders by Department",
        fontsize=14,
        fontweight="bold",
    )

    # Invert y-axis เพื่อให้อันดับ 1 อยู่บนสุด
    ax.invert_yaxis()

    # ปรับ x-axis เพื่อให้มีที่ว่างสำหรับ labels
    ax.set_xlim(0, dept_counts.max() * 1.25)

    plt.tight_layout()

    # บันทึกภาพ
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "categorical_distributions.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close()

    # แสดงสรุป
    print("\nสรุปการกระจายของ Department (Top 5):")
    for dept, count in dept_counts.head(5).items():
        pct = (count / total) * 100
        print(f"  • {dept}: {count:,} ({pct:.1f}%)")

    print(f"\n✅ บันทึกกราฟ categorical distributions: {output_path}")
    return output_path


def plot_correlation_matrix(
    df: pd.DataFrame,
    config: PlotConfig,
    output_dir: Path,
) -> tuple[Path, pd.DataFrame]:
    """
    สร้าง correlation matrix heatmap สำหรับ behavioral features

    Args:
        df: DataFrame ข้อมูล
        config: PlotConfig การตั้งค่า
        output_dir: โฟลเดอร์สำหรับบันทึกภาพ

    Returns:
        tuple ของ (Path ไปยังไฟล์ภาพ, DataFrame correlation matrix)
    """
    print("\n" + "=" * 60)
    print("สร้าง Correlation Matrix")
    print("=" * 60)

    # เลือกเฉพาะ behavioral features ที่มีในข้อมูล
    features = [f for f in config.behavioral_features if f in df.columns]

    if len(features) < 2:
        print("⚠️ ต้องการอย่างน้อย 2 features สำหรับ correlation matrix")
        return output_dir / "correlation_matrix.png", pd.DataFrame()

    print(f"คำนวณ correlation สำหรับ {len(features)} features...")

    # คำนวณ correlation matrix
    corr_matrix = df[features].corr()

    # สร้างกราฟ
    fig, ax = plt.subplots(figsize=config.figsize_correlation)

    # สร้าง mask สำหรับ upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=config.correlation_cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title(
        "Correlation Matrix of Behavioral Features",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    # บันทึกภาพ
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "correlation_matrix.png"
    plt.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close()

    # แสดง correlations ที่มีนัยสำคัญ (|r| > 0.3)
    print("\nCorrelations ที่มีค่ามากกว่า |0.3|:")
    significant_corrs: list[tuple[str, str, float]] = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.3:
                significant_corrs.append((features[i], features[j], corr_val))

    if significant_corrs:
        for f1, f2, corr in sorted(
            significant_corrs, key=lambda x: abs(x[2]), reverse=True
        ):
            direction = "positive" if corr > 0 else "negative"
            print(f"  • {f1} ↔ {f2}: {corr:.3f} ({direction})")
    else:
        print("  ไม่มี correlation ที่มีค่ามากกว่า |0.3|")

    print(f"\n✅ บันทึกกราฟ correlation matrix: {output_path}")
    return output_path, corr_matrix


def generate_all_visualizations(
    paths: DatasetPaths,
    config: PlotConfig,
) -> dict[str, Any]:
    """
    สร้างกราฟทั้งหมดและรวบรวมผลลัพธ์

    Args:
        paths: DatasetPaths สำหรับโหลดข้อมูลและบันทึกผลลัพธ์
        config: PlotConfig การตั้งค่า

    Returns:
        dict ที่มี paths และ results ทั้งหมด
    """
    print("\n" + "=" * 70)
    print(" 📊 สร้างกราฟวิเคราะห์ข้อมูล Instacart Customer Segmentation")
    print("=" * 70)

    # โหลดข้อมูล
    df = load_merged_data(paths)

    # สร้างโฟลเดอร์สำหรับ plots
    plots_dir = paths.output_dir / "plots"

    results: dict[str, Any] = {
        "data_shape": df.shape,
        "plots": {},
        "correlation_matrix": None,
    }

    # 1. Numeric distributions
    numeric_path = plot_numeric_distributions(df, config, plots_dir)
    results["plots"]["numeric"] = numeric_path

    # 2. Categorical distributions
    categorical_path = plot_categorical_distributions(df, config, plots_dir)
    results["plots"]["categorical"] = categorical_path

    # 3. Correlation matrix
    corr_path, corr_matrix = plot_correlation_matrix(df, config, plots_dir)
    results["plots"]["correlation"] = corr_path
    results["correlation_matrix"] = corr_matrix

    # สรุปผล
    print("\n" + "=" * 70)
    print(" ✅ สรุปผลการสร้างกราฟ")
    print("=" * 70)
    print(f"ข้อมูล: {results['data_shape'][0]:,} แถว, {results['data_shape'][1]} คอลัมน์")
    print(f"กราฟที่สร้าง:")
    for name, path in results["plots"].items():
        print(f"  • {name}: {path}")

    return results


def main() -> None:
    """Entry point สำหรับรันสคริปต์"""
    # กำหนด paths
    project_root = Path(__file__).resolve().parents[2]
    paths = DatasetPaths(
        data_dir=project_root / "data",
        output_dir=project_root / "reports" / "data_profile",
    )

    # กำหนด config
    config = PlotConfig()

    # สร้างกราฟทั้งหมด
    generate_all_visualizations(paths, config)


if __name__ == "__main__":
    main()
