import pandas as pd
import numpy as np
import joblib
import os
from src.config.config import load_config
from src.logger.logs import setup_logger

logger = setup_logger()

def predict(data):
    logger.info("Starting prediction pipeline")
    try:
        config = load_config()
        processed_dir = config['data']['processed_dir']
        model_dir = config['data'].get('model_dir', 'models')  # Corrected to use model_dir
        
        # Load model and preprocessing objects
        logger.info("Loading model and preprocessing objects")
        model_path = os.path.join(model_dir, 'best_model.pkl')  # Fixed file name
        scaler_path = os.path.join(processed_dir, 'scaler.pkl')
        pt_path = os.path.join(processed_dir, 'power_transformer.pkl')
        pca_path = os.path.join(processed_dir, 'pca.pkl')
        kmeans_path = os.path.join(processed_dir, 'kmeans.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        pt = joblib.load(pt_path)
        pca = joblib.load(pca_path)
        kmeans = joblib.load(kmeans_path)
        
        # Prepare input data
        logger.info("Preparing input data")
        df = pd.DataFrame([data])
        
        # Encode categorical variables
        month_revenue = {'Jan': 0.1, 'Feb': 0.1, 'Mar': 0.15, 'Apr': 0.12, 'May': 0.14, 'Jun': 0.13,
                         'Jul': 0.11, 'Aug': 0.09, 'Sep': 0.12, 'Oct': 0.13, 'Nov': 0.16, 'Dec': 0.15}
        df['Month_Encoded'] = df['Month'].map(month_revenue)
        
        visitor_freq = {'Returning_Visitor': 0.85, 'New_Visitor': 0.14, 'Other': 0.01}
        df['VisitorType_Freq'] = df['VisitorType'].map(visitor_freq)
        
        # Feature engineering
        logger.info("Engineering features")
        df['Total_PageViews'] = df['Administrative'] + df['Informational'] + df['ProductRelated']
        df['Total_Duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
        df['Avg_Time_Per_Page'] = df['Total_Duration'] / (df['Total_PageViews'] + 1)
        df['Bounce_Exit_Ratio'] = df['BounceRates'] / (df['ExitRates'] + 0.001)
        df['Engagement_Score'] = (df['PageValues'] * 0.5 + (1 - df['BounceRates']) * 0.3 + (1 - df['ExitRates']) * 0.2)
        
        # Handle skewed features
        logger.info("Transforming skewed features")
        skewed_features = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'PageValues']
        df[skewed_features] = pt.transform(df[skewed_features])
        
        # Select features
        final_features = [
            'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
            'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
            'SpecialDay', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend',
            'Month_Encoded', 'VisitorType_Freq', 'Total_PageViews', 'Total_Duration',
            'Avg_Time_Per_Page', 'Bounce_Exit_Ratio', 'Engagement_Score'
        ]
        
        X = df[final_features]
        
        # Handle missing values using training imputer
        logger.info("Handling missing values")
        imputer = joblib.load(os.path.join(processed_dir, 'imputer.pkl'))  # Load training imputer
        X = pd.DataFrame(imputer.transform(X), columns=X.columns)
        
        # Scale features
        logger.info("Scaling features")
        X_scaled = scaler.transform(X)
        
        # Apply PCA
        logger.info("Applying PCA")
        X_pca = pca.transform(X_scaled)
        
        # Apply clustering
        logger.info("Applying clustering")
        clusters = kmeans.predict(X_pca)
        X_final = np.hstack([X_pca, clusters.reshape(-1, 1)])
        
        # Predict
        logger.info("Making prediction")
        prediction = model.predict(X_final)[0]
        probability = model.predict_proba(X_final)[0][1]
        
        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
        return prediction, probability
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise Exception(f"Prediction failed: {str(e)}")