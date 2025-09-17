import os
import json
import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config.config import load_config
from src.logger.logs import setup_logger
from src.utils.data_loader import load_data
from src.data_processing.preprocess import preprocess_data

warnings.filterwarnings('ignore')


logger = setup_logger()

# Check for GPU availability
try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("GPU detected - checking for cuML")
except:
    GPU_AVAILABLE = False
    logger.info("No GPU detected - using CPU only")

# Check for cuML
if GPU_AVAILABLE:
    try:
        from cuml.ensemble import RandomForestClassifier as cuRF
        from cuml.svm import SVC as cuSVC
        CUML_AVAILABLE = True
        logger.info("cuML detected - using GPU-accelerated algorithms where available")
    except:
        CUML_AVAILABLE = False
        logger.info("cuML not available - using scikit-learn algorithms")
else:
    CUML_AVAILABLE = False

def train_model():
    logger.info("Starting model training pipeline (aligned with model_training (1).py)")
    try:
        config = load_config()
        processed_dir = config['data']['processed_dir']
        model_dir = config['data'].get('model_dir', 'models')
        
        # Regenerate preprocessed_data.csv
        logger.info("Regenerating preprocessed_data.csv automatically")
        preprocess_data()
        
        # Load preprocessed data
        preprocessed_path = os.path.join(processed_dir, 'preprocessed_data.csv')
        logger.info(f"Loading preprocessed data from {preprocessed_path}")
        df = load_data(preprocessed_path)
        
        # Separate train and test
        train_df = df[df['Set'] == 'Train'].drop('Set', axis=1)
        test_df = df[df['Set'] == 'Test'].drop('Set', axis=1)
        X_train = train_df.drop('Revenue', axis=1)
        y_train = train_df['Revenue']
        X_test = test_df.drop('Revenue', axis=1)
        y_test = test_df['Revenue']
        logger.info(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
        
        # Define models with GPU support where available
        models = {
            'XGBoost': {
                'model': XGBClassifier(
                    random_state=42, 
                    eval_metric='logloss',
                    tree_method='gpu_hist' if GPU_AVAILABLE else 'auto',
                    gpu_id=0 if GPU_AVAILABLE else -1,
                    verbosity=0  # Reduce output verbosity
                ),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [3, 4, 5],
                    'classifier__learning_rate': [0.05, 0.01],
                    'classifier__subsample': [0.8, 0.9],
                    'classifier__colsample_bytree': [0.8, 0.9],
                    'classifier__gamma': [0, 0.1, 0.2]
                }
            },
            'LightGBM': {
                'model': LGBMClassifier(
                    random_state=42, 
                    class_weight='balanced',
                    device='gpu' if GPU_AVAILABLE else 'cpu',
                    verbose=-1  # Silence output
                ),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__learning_rate': [0.05, 0.01],
                    'classifier__num_leaves': [31, 50, 70],
                    'classifier__subsample': [0.8, 0.9],
                    'classifier__colsample_bytree': [0.8, 0.9]
                }
            }
        }

        # Add GPU-accelerated models if available
        if CUML_AVAILABLE:
            models['RandomForest_GPU'] = {
                'model': cuRF(
                    random_state=42,
                    class_weight='balanced'
                ),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [5, 10, 15],
                    'classifier__max_features': [0.5, 0.7, 0.9]
                }
            }
            
            models['SVM_GPU'] = {
                'model': cuSVC(
                    random_state=42,
                    probability=True
                ),
                'params': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__kernel': [0, 1],  # 0=linear, 1=rbf in cuML
                    'classifier__gamma': [0.1,'auto', 1, 10]
                }
            }
        else:
            # Add CPU versions if GPU not available
            models['RandomForest'] = {
                'model': RandomForestClassifier(
                    random_state=42, 
                    class_weight='balanced',
                    n_jobs=-1  # Use all cores
                ),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [5, 7, 10,None],
                    'classifier__min_samples_split': [2, 5],
                    'classifier__min_samples_leaf': [1, 2],
                    'classifier__max_features': [0.5, 0.7, 0.9],
                    'classifier__bootstrap': [True, False]
                    
                }
            }
            
            models['GradientBoosting'] = {
                'model': GradientBoostingClassifier(
                    random_state=42,
                    verbose=0  # Silence output
                ),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__learning_rate': [0.05, 0.01],
                    'classifier__max_depth': [3, 4, 5],
                    'classifier__subsample': [0.8, 0.9],
                    'classifier__min_samples_split': [2, 5, 10]
                }
            }

        # For even faster training, we can use a subset of data for hyperparameter tuning
        # This is optional but can significantly speed up the process
        use_subset_for_tuning = True
        tuning_subset_size = 0.5  # Use 50% of data for tuning

        if use_subset_for_tuning:
            from sklearn.utils import resample
            X_tune, _, y_tune, _ = train_test_split(
                X_train, y_train, 
                train_size=tuning_subset_size, 
                random_state=42, 
                stratify=y_train
            )
            print(f"Using {tuning_subset_size*100}% of data ({len(X_tune)} samples) for hyperparameter tuning")
        else:
            X_tune, y_tune = X_train, y_train

        # Perform hyperparameter tuning and evaluation
        results = {}
        best_models = {}

        for name, model_info in models.items():
            print(f"\nTraining and tuning {name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('classifier', model_info['model'])
            ])
            
            # Use Bayesian Optimization for more efficient hyperparameter search
            # If scikit-optimize is available
            try:
                from skopt import BayesSearchCV
                from skopt.space import Real, Categorical, Integer
                
                # Convert parameter grid to skopt space
                param_space = {}
                for param, values in model_info['params'].items():
                    if all(isinstance(v, (int, np.integer)) for v in values):
                        param_space[param] = Integer(min(values), max(values))
                    elif all(isinstance(v, (float, np.floating)) for v in values):
                        param_space[param] = Real(min(values), max(values))
                    else:
                        param_space[param] = Categorical(values)
                
                search = BayesSearchCV(
                    pipeline,
                    param_space,
                    n_iter=10,  # Fewer iterations needed with Bayesian optimization
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                    scoring='f1',
                    n_jobs=1 if CUML_AVAILABLE else -1,  # cuML doesn't support n_jobs > 1
                    random_state=42,
                    verbose=0
                )
                method = "Bayesian Optimization"
                
            except ImportError:
                # Fall back to RandomizedSearchCV if scikit-optimize not available
                search = RandomizedSearchCV(
                    pipeline, 
                    model_info['params'], 
                    n_iter=10,
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                    scoring='f1',
                    n_jobs=1 if CUML_AVAILABLE else -1,
                    random_state=42,
                    verbose=0
                )
                method = "Randomized Search"
            
            print(f"Using {method} for hyperparameter tuning")
            
            # Fit the model
            search.fit(X_tune, y_tune)
            
            # Get the best model
            best_model = search.best_estimator_
            best_models[name] = best_model
            
            # Make predictions on full test set
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            
            # Store results
            results[name] = {
                'best_params': search.best_params_,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'cv_score': search.best_score_
            }
            
            print(f"Best parameters: {search.best_params_}")
            print(f"F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

            logger.info(f"{name} - Best F1 Score: {search.best_score_:.4f}, Test F1 Score: {results[name]['f1']:.4f}")
        
        # Model comparison
        results_df = pd.DataFrame(results).T[['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'cv_score']]
        results_df = results_df.sort_values('f1', ascending=False)
        logger.info("\nModel Comparison:\n" + results_df.to_string())
        
        # Best model
        best_model_name = results_df.index[0]
        best_model = best_models[best_model_name]
        logger.info(f"\nBest model: {best_model_name}")
        
        # Detailed evaluation
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        logger.info(f"Precision: {precision_score(y_test, y_pred):.4f}")
        logger.info(f"Recall: {recall_score(y_test, y_pred):.4f}")
        logger.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        logger.info(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Overfitting check
        train_pred = best_model.predict(X_train)
        train_prob = best_model.predict_proba(X_train)[:, 1]
        logger.info("\nOverfitting Check:")
        logger.info(f"Train F1 Score: {f1_score(y_train, train_pred):.4f}")
        logger.info(f"Test F1 Score: {f1_score(y_test, y_pred):.4f}")
        logger.info(f"Train ROC AUC: {roc_auc_score(y_train, train_prob):.4f}")
        logger.info(f"Test ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
        
        # Save best model
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'best_model.pkl')
        joblib.dump(best_model, model_path)
        logger.info(f"Best model saved to {model_path}")
        
        # Generate evaluation plots
        generate_evaluation_plots(y_test, y_pred, y_prob, best_model_name, X_test, best_model)
        
        logger.info("Model training completed successfully")
        return best_model
    
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}", exc_info=True)
        raise


def generate_evaluation_plots(y_test, y_pred, y_prob, model_name, X_test, model):
    """
    Generate static evaluation plots using Matplotlib and Seaborn, saving them as PNG files.
    
    Parameters:
    - y_test: True binary labels (0 or 1)
    - y_pred: Predicted binary labels
    - y_prob: Predicted probabilities for the positive class
    - model_name: Name of the model (string)
    - X_test: Test features (used for feature names in feature importance)
    - model: Trained model object (scikit-learn pipeline or classifier)
    """
    logger.info("Generating evaluation plots (model metrics)")
    try:
        # Create the output directory if it doesn't exist
        plots_dir = 'static/plots'
        os.makedirs(plots_dir, exist_ok=True)

        # Plot 1: Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Revenue', 'Revenue'],
                    yticklabels=['No Revenue', 'Revenue'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{model_name}.png'))
        plt.close()
        logger.info(f"Saved confusion matrix plot to {plots_dir}/confusion_matrix_{model_name}.png")

        # Plot 2: ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'roc_curve_{model_name}.png'))
        plt.close()
        logger.info(f"Saved ROC curve plot to {plots_dir}/roc_curve_{model_name}.png")

        # Plot 3: Feature Importance (for tree-based models)
        if hasattr(model.named_steps.get('classifier', model), 'feature_importances_'):
            logger.info("Generating feature importance plot")
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.named_steps['classifier'].feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
            plt.title(f'Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'feature_importance_{model_name}.png'))
            plt.close()
            logger.info(f"Saved feature importance plot to {plots_dir}/feature_importance_{model_name}.png")
        else:
            logger.info("Skipping feature importance plot: Model does not support feature_importances_")

    except Exception as e:
        logger.error(f"Failed to generate evaluation plots: {str(e)}", exc_info=True)
        raise
    
if __name__ == "__main__":
    train_model()