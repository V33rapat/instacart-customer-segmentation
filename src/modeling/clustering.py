# Importing library
from zipfile import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from kneed import KneeLocator

def perform_clustering(data, max_clusters=10, batch_size=100):
    # สเกลข้อมูลด้วย StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.drop(columns=['user_id', 'cluster']))
    print('\nData standardized.')
    print(data_scaled)

    # ลดมิติข้อมูลด้วย PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    # ค้นหาจำนวนคลัสเตอร์ที่เหมาะสมด้วย Elbow Method
    print('\nElbow Method:')
    inertia = []
    for n_clusters in range(1, max_clusters + 1):
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(data_scaled)
        inertia.append(model.inertia_)
        print(f'Fitted KMeans with {n_clusters} clusters, Inertia: {model.inertia_}')

    plot_elbow_method(inertia, max_clusters)

    # หา elbow point ด้วย KneeLocator
    kl = KneeLocator(range(1, max_clusters + 1), inertia, curve="convex", direction="decreasing")
    optimal_clusters = kl.elbow
    print(f'\nOptimal number of clusters determined by KneeLocator: {optimal_clusters}')
    
    # หา k ด้วย Silhouette Score โดยใช้ MiniBatchKMeans
    print('\nSilhouette Score Method:')
    silhouette_avg = []
    for n_clusters in range(2, max_clusters + 1):
        mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42)
        cluster_labels = mbk.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, cluster_labels, sample_size=10000, random_state=42)
        silhouette_avg.append(score)
        print(f'Fitted MiniBatchKMeans with {n_clusters} clusters, Silhouette Score: {silhouette_avg[-1]}')

    # สร้างโมเดล KMeans ด้วยจำนวนคลัสเตอร์ที่เหมาะสม
    final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    cluster_labels = final_kmeans.fit_predict(data_pca)

    return cluster_labels, optimal_clusters, silhouette_avg, data_pca

def plot_clusters(data_pca, cluster_labels):
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=cluster_labels, cmap='viridis', s=10)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Customer Segments based on PCA')
    plt.show()

def plot_elbow_method(inertia, max_clusters):
    plt.plot(range(1, max_clusters + 1), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()

if __name__ == "__main__":
    # Example usage
    print('Loading data...')
    df = pd.read_csv('feature/user_features.csv')  # Load your data here
    print('Data loaded successfully.')
    print(df.head())
    print('\nInfomation data')
    print(df.info())
    print('\nData description')
    print(df.describe())

    cluster_labels, optimal_clusters, silhouette_avg, data_pca = perform_clustering(df)

    print(f'Optimal number of clusters: {optimal_clusters}')
    print(f'Silhouette Score: {silhouette_avg}')

    plot_clusters(data_pca, cluster_labels)

    df_success = df.copy()
    df_success['cluster'] = cluster_labels
    
    print('\nCluster assignments added to the dataframe.')
    print(df_success.head())

    df_grouped = df_success.groupby('cluster').mean()
    print('\nCluster-wise mean of features:')
    print(df_grouped)

    # Save the clustered data to a new CSV file
    # Create results directory if it doesn't exist
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    df_grouped.to_csv('results/clustered_data.csv', index=False)
    print('\nClustered data saved to results/clustered_data.csv')
    