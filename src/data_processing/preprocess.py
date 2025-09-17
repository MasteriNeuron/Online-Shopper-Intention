import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os
from src.config.config import load_config
from src.logger.logs import setup_logger
from src.utils.data_loader import save_data

logger = setup_logger()

def preprocess_data():
    logger.info("Starting data preprocessing")
    try:
        config = load_config()
        raw_url = config['data']['raw_url']
        processed_dir = config['data']['processed_dir']
        
        # Load dataset from URL
        logger.info(f"Loading data from {raw_url}")
        df = pd.read_csv(raw_url)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        # Check and remove duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.info(f"Removing {duplicates} duplicate rows")
            df = df.drop_duplicates()
            logger.info(f"Data shape after removing duplicates: {df.shape}")
        
        # Convert boolean columns to int
        logger.info("Converting boolean columns to integers")
        df['Revenue'] = df['Revenue'].astype(int)
        df['Weekend'] = df['Weekend'].astype(int)
        
        # Feature engineering before splitting (no target leakage)
        logger.info("Creating interaction features")
        df['Total_PageViews'] = df['Administrative'] + df['Informational'] + df['ProductRelated']
        df['Total_Duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
        df['Avg_Time_Per_Page'] = df['Total_Duration'] / (df['Total_PageViews'] + 1)
        df['Bounce_Exit_Ratio'] = df['BounceRates'] / (df['ExitRates'] + 0.001)
        df['Engagement_Score'] = (df['PageValues'] * 0.5 +
                                 (1 - df['BounceRates']) * 0.3 +
                                 (1 - df['ExitRates']) * 0.2)
        
        # Handle categorical variables with advanced encoding
        logger.info("Applying target encoding for Month")
        month_revenue = df.groupby('Month')['Revenue'].mean().to_dict()
        df['Month_Encoded'] = df['Month'].map(month_revenue)
        
        logger.info("Applying frequency encoding for VisitorType")
        visitor_freq = df['VisitorType'].value_counts(normalize=True).to_dict()
        df['VisitorType_Freq'] = df['VisitorType'].map(visitor_freq)
        
        # Handle skewed features
        logger.info("Transforming skewed features")
        skewed_features = ['Administrative_Duration', 'Informational_Duration',
                           'ProductRelated_Duration', 'PageValues']
        pt = PowerTransformer(method='yeo-johnson')
        df[skewed_features] = pt.fit_transform(df[skewed_features])
        
        # Select features for modeling
        numeric_features = [
            'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
            'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
            'SpecialDay', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend',
            'Month_Encoded', 'VisitorType_Freq', 'Total_PageViews', 'Total_Duration',
            'Avg_Time_Per_Page', 'Bounce_Exit_Ratio', 'Engagement_Score'
        ]
        
        X = df[numeric_features]
        y = df['Revenue']
        
        # Handle class imbalance with SMOTE
        logger.info("Applying SMOTE to handle class imbalance")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )
        
        # Scaling
        logger.info("Scaling features")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # PCA
        logger.info("Applying PCA for dimensionality reduction")
        pca = PCA(n_components=0.95, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        logger.info(f"PCA reduced features to: {X_train_pca.shape[1]} components")
        
        # K-Means Clustering
        logger.info("Applying K-Means clustering")
        k_range = range(2, 10)
        sil_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_train_pca)
            sil_scores.append(silhouette_score(X_train_pca, kmeans.labels_))
        
        optimal_k = k_range[np.argmax(sil_scores)]
        logger.info(f"Optimal number of clusters: {optimal_k}")
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        train_clusters = kmeans.fit_predict(X_train_pca)
        test_clusters = kmeans.predict(X_test_pca)
        
        # Combine features
        logger.info("Combining PCA features with cluster labels")
        X_train_final = np.hstack([X_train_pca, train_clusters.reshape(-1, 1)])
        X_test_final = np.hstack([X_test_pca, test_clusters.reshape(-1, 1)])
        
        # Save preprocessed data
        logger.info("Saving preprocessed data")
        train_df = pd.DataFrame(X_train_final, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])] + ['Cluster'])
        train_df['Revenue'] = y_train.reset_index(drop=True)
        train_df['Set'] = 'Train'
        
        test_df = pd.DataFrame(X_test_final, columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])] + ['Cluster'])
        test_df['Revenue'] = y_test.reset_index(drop=True)
        test_df['Set'] = 'Test'
        
        preprocessed_df = pd.concat([train_df, test_df], ignore_index=True)
        preprocessed_path = os.path.join(processed_dir, 'preprocessed_data.csv')
        save_data(preprocessed_df, preprocessed_path)
        
        # Save models
        pt_path = os.path.join(processed_dir, 'power_transformer.pkl')
        joblib.dump(pt, pt_path)
        logger.info(f"PowerTransformer saved to {pt_path}")
        
        scaler_path = os.path.join(processed_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        pca_path = os.path.join(processed_dir, 'pca.pkl')
        joblib.dump(pca, pca_path)
        logger.info(f"PCA model saved to {pca_path}")
        
        kmeans_path = os.path.join(processed_dir, 'kmeans.pkl')
        joblib.dump(kmeans, kmeans_path)
        logger.info(f"K-Means model saved to {kmeans_path}")
        
        # Save month encoding mapping
        joblib.dump(month_revenue, os.path.join(processed_dir, 'month_encoding_mapping.pkl'))

        # Save visitor type frequency mapping
        joblib.dump(visitor_freq, os.path.join(processed_dir, 'visitor_type_freq_mapping.pkl'))
        # Save clean data
        clean_path = os.path.join(processed_dir, 'clean.csv')
        save_data(df, clean_path)
        
        logger.info("Data preprocessing completed successfully")
        return X_train_final, X_test_final, y_train, y_test
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    preprocess_data()