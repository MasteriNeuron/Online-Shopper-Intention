from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
from src.utils.dash_app import create_dash_app  # Import the Dash app
import joblib
from src.data_processing.preprocess import preprocess_data
from src.utils.data_loader import save_data
from src.logger.logs import setup_logger
from src.pipelines.training_pipeline import train_model, generate_evaluation_plots
import json

# ----------------- Flask App -----------------
app = Flask(__name__, template_folder="templates", static_folder="static")
logger = setup_logger()

# Initialize Dash app
create_dash_app(app)  # Mount Dash app at /eda/
logger.info("Dash app initialized at /eda/")
# ----------------- Config & Artifacts -----------------
# Global variable to hold the trained model
trained_model = None

# ---------- Helpers ----------
REQUIRED_INPUT_KEYS = [
    'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
    'SpecialDay', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend',
    'Month', 'VisitorType'
]

def coerce_single_payload_from_request(req):
    """
    Accept both JSON and HTML form posts, normalize to the dict your pipeline expects.
    """
    if req.is_json:
        src = req.get_json() or {}
        out = {}
        for k in REQUIRED_INPUT_KEYS:
            if k in src:
                out[k] = src[k]
    else:
        f = req.form
        out = {
            'Administrative': float(f.get('Administrative', 0)),
            'Administrative_Duration': float(f.get('Administrative_Duration', 0)),
            'Informational': float(f.get('Informational', 0)),
            'Informational_Duration': float(f.get('Informational_Duration', 0)),
            'ProductRelated': float(f.get('ProductRelated', 0)),
            'ProductRelated_Duration': float(f.get('ProductRelated_Duration', 0)),
            'BounceRates': float(f.get('BounceRates', 0)),
            'ExitRates': float(f.get('ExitRates', 0)),
            'PageValues': float(f.get('PageValues', 0)),
            'SpecialDay': float(f.get('SpecialDay', 0)),
            'OperatingSystems': int(f.get('OperatingSystems', 0)),
            'Browser': int(f.get('Browser', 0)),
            'Region': int(f.get('Region', 0)),
            'TrafficType': int(f.get('TrafficType', 0)),
            'Weekend': int(f.get('Weekend', 0)),
            'Month': f.get('Month', ''),
            'VisitorType': f.get('VisitorType', '')
        }
    
    # Convert to appropriate types
    numeric_keys = [
        'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
        'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
        'SpecialDay'
    ]
    
    int_keys = ['OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend']
    
    for k in numeric_keys:
        if k in out:
            try:
                out[k] = float(out[k])
            except Exception:
                out[k] = 0.0
                
    for k in int_keys:
        if k in out:
            try:
                out[k] = int(out[k])
            except Exception:
                out[k] = 0
                
    return out

def preprocess_single_instance(data, preprocessor_path='datasets/processed'):
    """
    Preprocess a single instance for prediction using saved preprocessors
    """
    try:
        # Load preprocessors
        pt = joblib.load(os.path.join(preprocessor_path, 'power_transformer.pkl'))
        scaler = joblib.load(os.path.join(preprocessor_path, 'scaler.pkl'))
        pca = joblib.load(os.path.join(preprocessor_path, 'pca.pkl'))
        kmeans = joblib.load(os.path.join(preprocessor_path, 'kmeans.pkl'))
        
        # Load month encoding and visitor type frequency mappings
        month_revenue = joblib.load(os.path.join(preprocessor_path, 'month_encoding_mapping.pkl'))
        visitor_freq = joblib.load(os.path.join(preprocessor_path, 'visitor_type_freq_mapping.pkl'))
        
        # Create DataFrame
        df = pd.DataFrame([data])
        logger.info(f"Preprocessing single instance: {data}")
        
        # Convert boolean columns to int
        df['Weekend'] = df['Weekend'].astype(int)
        
        # Apply encodings using the saved mappings
        df['Month_Encoded'] = df['Month'].map(month_revenue)
        df['VisitorType_Freq'] = df['VisitorType'].map(visitor_freq)
        
        # Feature engineering
        df['Total_PageViews'] = df['Administrative'] + df['Informational'] + df['ProductRelated']
        df['Total_Duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
        df['Avg_Time_Per_Page'] = df['Total_Duration'] / (df['Total_PageViews'] + 1)
        df['Bounce_Exit_Ratio'] = df['BounceRates'] / (df['ExitRates'] + 0.001)
        df['Engagement_Score'] = (df['PageValues'] * 0.5 +
                                 (1 - df['BounceRates']) * 0.3 +
                                 (1 - df['ExitRates']) * 0.2)
        
        # Handle skewed features
        skewed_features = ['Administrative_Duration', 'Informational_Duration',
                          'ProductRelated_Duration', 'PageValues']
        df[skewed_features] = pt.transform(df[skewed_features])
        
        # Select features for modeling
        numeric_features = [
            'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
            'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
            'SpecialDay', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend',
            'Month_Encoded', 'VisitorType_Freq', 'Total_PageViews', 'Total_Duration',
            'Avg_Time_Per_Page', 'Bounce_Exit_Ratio', 'Engagement_Score'
        ]
        
        X = df[numeric_features]
        logger.info(f"Features before scaling and PCA: {X.to_dict(orient='records')[0]}")
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Apply PCA
        X_pca = pca.transform(X_scaled)
        
        # Apply KMeans clustering
        cluster = kmeans.predict(X_pca)
        
        # Combine features
        X_final = np.hstack([X_pca, cluster.reshape(-1, 1)])
        
        return X_final
        
    except Exception as e:
        logger.error(f"Error in preprocessing single instance: {str(e)}")
        raise

def predict_single_instance(data, model, preprocessor_path='datasets/processed'):
    """
    Predict a single instance
    """
    try:
        # Preprocess the data
        X_final = preprocess_single_instance(data, preprocessor_path)
        
        # Make prediction
        prediction = model.predict(X_final)
        probability = model.predict_proba(X_final)
        
        return prediction[0], probability[0]
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

def predict_batch(df, model, preprocessor_path='datasets/processed'):
    """
    Perform batch predictions
    """
    try:
        predictions = []
        probabilities = []
        
        for _, row in df.iterrows():
            data = row.to_dict()
            pred, prob = predict_single_instance(data, model, preprocessor_path)
            predictions.append(pred)
            probabilities.append(prob)
            
        result_df = df.copy()
        result_df['Prediction'] = predictions
        result_df['Probability_0'] = [p[0] for p in probabilities]
        result_df['Probability_1'] = [p[1] for p in probabilities]
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise

def save_predictions(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

# ----------------- Routes -----------------
@app.route('/')
def root():
    logger.info("GET / → redirect to /home")
    return render_template('home.html')

@app.route('/home')
def home():
    logger.info("GET /home → home.html")
    return render_template('home.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        logger.info("GET /train → train.html")
        return render_template('train.html')
    elif request.method == 'POST':
        logger.info("POST /train")
        try:
            # Initialize log for training steps
            training_log = []
            
            # Log: Start training
            training_log.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Initializing training process...")
            logger.info("Initializing training process...")
            
            # Run the training pipeline
            training_log.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Running training pipeline...")
            logger.info("Running training pipeline...")
            
            global trained_model
            trained_model = train_model()
            
            # Log: Training completed
            training_log.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Training completed successfully")
            logger.info("Training completed successfully")
            
            msg = "Training completed successfully"
            logger.info(msg)
            return jsonify({'status': 'success', 'message': msg, 'logs': training_log})
            
        except Exception as e:
            logger.exception("Training failed")
            training_log.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Error: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e), 'logs': training_log}), 500
        

@app.route('/eda')
def eda():
    logger.info("GET /eda → rendering eda.html")
    try:
        csv_path = os.path.join('datasets', 'processed', 'clean.csv')
        if not os.path.exists(csv_path):
            logger.error(f"clean.csv not found at {csv_path}")
            return render_template('eda.html', error="Data file not found. Please ensure clean.csv is available.")
        return render_template('eda.html')
    except Exception as e:
        logger.error(f"Error rendering /eda: {str(e)}")
        return render_template('eda.html', error=str(e))

@app.route('/predict', methods=['GET', 'POST'])
def predict_route():
    global trained_model  # Declare global at the start
    if request.method == 'GET':
        logger.info("GET /predict → predict.html")
        return render_template('predict.html')
    elif request.method == 'POST':
        logger.info("POST /predict")
        try:
            if trained_model is None:
                try:
                    model_path = os.path.join('models', 'best_model.pkl')
                    trained_model = joblib.load(model_path)
                    logger.info("Loaded trained model from disk")
                except:
                    return jsonify({'status': 'error', 'message': 'Model not trained yet. Please train the model first.'}), 400
            data = coerce_single_payload_from_request(request)
            logger.info(f"Predict payload: {data}")
            prediction, probability = predict_single_instance(data, trained_model)
            logger.info(f"Prediction: {prediction}, Probability: {probability}")
            result = {
                'status': 'success', 
                'prediction': int(prediction),
                'probability_0': float(probability[0]),
                'probability_1': float(probability[1]),
                'revenue_prediction': "Yes" if prediction == 1 else "No"
            }
            return jsonify(result)
        except Exception as e:
            logger.exception("Prediction failed")
            return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict_route():
    global trained_model  # Declare global at the start
    logger.info("POST /batch_predict")
    try:
        if trained_model is None:
            try:
                model_path = os.path.join('models', 'best_model.pkl')
                trained_model = joblib.load(model_path)
                logger.info("Loaded trained model from disk")
            except:
                return jsonify({'status': 'error', 'message': 'Model not trained yet. Please train the model first.'}), 400
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided.'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected.'}), 400
        os.makedirs('temp', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        filename = secure_filename(file.filename)
        temp_path = os.path.join('temp', filename)
        file.save(temp_path)
        df = pd.read_csv(temp_path)
        logger.info(f"Uploaded CSV shape: {df.shape}")
        result_df = predict_batch(df, trained_model)
        out_name = f'predictions_{filename}'
        output_path = os.path.join('output', out_name)
        save_predictions(result_df, output_path)
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return jsonify({
            'status': 'success',
            'message': f'Batch prediction completed. {len(result_df)} rows processed.',
            'download_link': f'/download/{out_name}'
        })
    except Exception as e:
        logger.exception("Batch prediction failed")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """Serve files from /output for download."""
    return send_from_directory('output', filename, as_attachment=True)

@app.route('/logs', methods=['GET'])
def get_logs():
    """Fetch all logs from logs/pipeline.log."""
    try:
        log_file_path = os.path.join('logs', 'pipeline.log')
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r', encoding='utf-8', errors='replace') as f:
                logs = f.readlines()
            return jsonify({'status': 'success', 'logs': [log.strip() for log in logs]})
        else:
            logger.warning("Log file logs/pipeline.log not found")
            return jsonify({'status': 'error', 'message': 'Log file not found', 'logs': []}), 404
    except Exception as e:
        logger.error(f"Failed to read logs: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e), 'logs': []}), 500

# ----------------- Entrypoint -----------------
if __name__ == '__main__':
    # Try to load the model at startup if it exists
    try:
        model_path = os.path.join('models', 'best_model.pkl')
        trained_model = joblib.load(model_path)
        logger.info("Model loaded successfully at startup")
    except Exception as e:
        logger.warning(f"Could not load model at startup: {str(e)}")
        trained_model = None
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)