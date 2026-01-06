import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
try:
    import shap
except ImportError:
    shap = None

from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import json
import logging
from datetime import datetime
import uuid

from .config import settings
from .db import db
from .data import load_training_data

log = logging.getLogger("training.core")

ALGORITHMS = {
    "linear_regression": LinearRegression,
    "random_forest_regressor": RandomForestRegressor,
    "logistic_classification": LogisticRegression,
    "random_forest_classifier": RandomForestClassifier
}

def train_model_task(training_id: str, symbol: str, algorithm: str, target_col: str, params: dict, data_options: str = None):
    model_path = str(settings.models_dir / f"{training_id}.joblib")
    
    try:
        log.info(f"Starting training {training_id} for {symbol} using {algorithm}")
        
        # 1. Load Data
        df = load_training_data(symbol, target_col=target_col, options_filter=data_options)
        
        # 1a. Detect Split Column (Train/Test)
        split_col_name = None
        for col in df.columns:
            if df[col].dtype == object or isinstance(df[col].dtype, pd.CategoricalDtype):
                unique_vals = set(df[col].dropna().astype(str).str.lower().unique())
                if "train" in unique_vals and "test" in unique_vals:
                    split_col_name = col
                    log.info(f"Found existing split column: {split_col_name}")
                    break
        
        # 2. Prepare Features
        # Drop metadata columns and target
        drop_cols = ["target", "ts", "symbol", "date", "source", "options", "target_col_shifted"]
        if split_col_name:
            drop_cols.append(split_col_name)
            
        # Filter for numeric columns only to be safe
        df_numeric = df.select_dtypes(include=[np.number])
        feature_cols = [c for c in df_numeric.columns if c not in drop_cols]
        
        if not feature_cols:
            raise ValueError("No numeric feature columns found for training")

        # 1. Drop rows where Target is missing (we cannot train without a label)
        rows_before = len(df)
        df = df.dropna(subset=["target"])
        if df.empty:
            raise ValueError(f"Dataset empty after dropping missing targets. Input rows: {rows_before}")

        X = df[feature_cols]
        y = df["target"]
        
        # 2. Handle Feature NaNs
        # First, drop feature columns that are 100% NaN (cannot impute them)
        initial_feature_cols = list(X.columns)
        X = X.dropna(axis=1, how='all')
        if X.empty:
            raise ValueError("All feature columns were entirely NaN.")

        feature_cols_used = list(X.columns)
        dropped_feature_cols = sorted(list(set(initial_feature_cols) - set(feature_cols_used)))
        
        X = X[feature_cols_used] # Update X to only valid cols

        # If Classifier, convert target to binary direction (1 = Up/Same, 0 = Down)
        # We compare Future Target (df['target']) vs Current Price (df[target_col])
        if "classifier" in algorithm or "classification" in algorithm:
            log.info(f"Converting target to binary direction for classifier {algorithm}")
            # Ensure target_col exists in original df for comparison
            # Note: df still matches indices of X and y
            if target_col in df.columns:
                # Handle possible NaN in current price (unlikely if strictly filtered, but possible)
                if df[target_col].isna().any():
                    # For safety, drop rows where current price is NaN (needed for classification logic)
                    valid_indices = df[target_col].notna()
                    df = df[valid_indices]
                    X = X[valid_indices]
                    y = y[valid_indices]
                
                y = (df["target"] >= df[target_col]).astype(int)
            else:
                 log.warning(f"Target column {target_col} not found for direction calc. Using > 0 check.")
                 y = (df["target"] > 0).astype(int)

        # Split Data (Custom Column or Time-based)
        if split_col_name:
            log.info(f"Splitting data using column: {split_col_name}")
            # Aligned split values
            split_vals = df.loc[X.index, split_col_name].astype(str).str.lower()
            
            train_mask = split_vals == 'train'
            test_mask = split_vals == 'test'
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            
            if X_train.empty or X_test.empty:
                log.warning(f"Split column resulted in empty train/test. Train: {len(X_train)}, Test: {len(X_test)}. Falling back to time-based.")
                split_col_name = None # Trigger fallback

        if not split_col_name:
            # Simple time-based split (80/20)
            split_idx = int(len(X) * 0.8)
            
            # Split
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 3. Impute Missing Values in Features (Mean Strategy)
        # This fills gaps (e.g. from indicators) without dropping data
        if X_train.isna().any().any() or X_test.isna().any().any():
            log.info("Imputing missing feature values with mean")
            imputer = SimpleImputer(strategy='mean')
            # Fit on train, transform both
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            # Note: X_train/X_test are now numpy arrays, which SKLearn accepts fine
            
        # 3. Initialize Model
        ModelClass = ALGORITHMS.get(algorithm)
        if not ModelClass:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        model = ModelClass(**params)
        
        # 4. Train
        model.fit(X_train, y_train)
        
        # 5. Evaluate
        preds = model.predict(X_test)
        
        metrics = {}
        if dropped_feature_cols:
            metrics["dropped_cols"] = dropped_feature_cols
        metrics["features_count"] = len(feature_cols_used)
        
        if "regressor" in algorithm or "regression" in algorithm:
            mse = mean_squared_error(y_test, preds)
            metrics["mse"] = mse
            metrics["rmse"] = float(mse ** 0.5)
        else:
            acc = accuracy_score(y_test, preds)
            metrics["accuracy"] = acc

        # Initialize detailed feature info
        feature_details = {col: {} for col in feature_cols_used}

        # 1. Linear Coefficients / Tree Importance
        try:
            if hasattr(model, "feature_importances_"):
                imps = model.feature_importances_
                for i, col in enumerate(feature_cols_used):
                    feature_details[col]["tree_importance"] = float(imps[i])
            elif hasattr(model, "coef_"):
                coefs = model.coef_
                if coefs.ndim > 1: coefs = coefs[0]
                for i, col in enumerate(feature_cols_used):
                    feature_details[col]["coefficient"] = float(coefs[i])
        except Exception as e:
            log.warning(f"Error extracting base importance: {e}")

        # 2. Permutation Importance (Model Agnostic)
        try:
            # Use specific scorer based on task
            scorer = 'neg_mean_squared_error' if ("regressor" in algorithm or "regression" in algorithm) else 'accuracy'
            # Use a sample for speed if needed
            X_perm = X_test[:500]
            y_perm = y_test[:500]
            if len(X_perm) > 10:
                perm_result = permutation_importance(model, X_perm, y_perm, n_repeats=5, random_state=42, scoring=scorer)
                for i, col in enumerate(feature_cols_used):
                    feature_details[col]["permutation_mean"] = float(perm_result.importances_mean[i])
        except Exception as e:
             log.warning(f"Error extracting permutation importance: {e}")

        # 3. SHAP Values
        if shap:
            try:
                # Use a small background sample for explainers
                X_bg = X_train[:100]
                X_eval = X_test[:100]
                
                explainer = None
                if "Linear" in str(type(model)) or "Logistic" in str(type(model)):
                     explainer = shap.LinearExplainer(model, X_bg)
                elif "Forest" in str(type(model)) or "Tree" in str(type(model)):
                     explainer = shap.TreeExplainer(model)
                
                if explainer and len(X_eval) > 0:
                     shap_vals = explainer.shap_values(X_eval)
                     # For classification, shap_vals might be a list of arrays (one per class). Take the first (or positive class)
                     if isinstance(shap_vals, list):
                         shap_vals = shap_vals[-1] # Positive class
                         
                     # Mean Absolute SHAP value per feature
                     shap_mean = np.mean(np.abs(shap_vals), axis=0)
                     for i, col in enumerate(feature_cols_used):
                         feature_details[col]["shap_mean_abs"] = float(shap_mean[i])
            except Exception as e:
                log.warning(f"Error extracting SHAP values: {e}")

        metrics["feature_details"] = feature_details
        
        # Legacy/Simple Importance (Sort by best available metric for the simple list)
        # Priority: SHAP > Tree > Permutation > Coeff
        legacy_imp = {}
        for col, dets in feature_details.items():
            val = 0
            if "shap_mean_abs" in dets: val = dets["shap_mean_abs"]
            elif "tree_importance" in dets: val = dets["tree_importance"]
            elif "permutation_mean" in dets: val = dets["permutation_mean"]
            elif "coefficient" in dets: val = abs(dets["coefficient"])
            legacy_imp[col] = val
        
        if legacy_imp:
             # Sort by absolute value descending
             metrics["feature_importance"] = dict(sorted(legacy_imp.items(), key=lambda item: abs(item[1]), reverse=True))

        # 6. Save
        joblib.dump(model, model_path)
        
        # 7. Update DB
        db.update_model_status(
            training_id, 
            status="completed", 
            metrics=json.dumps(metrics), 
            artifact_path=model_path,
            feature_cols=json.dumps(feature_cols_used)
        )
        log.info(f"Training {training_id} completed. Metrics: {metrics}")
        
    except Exception as e:
        log.exception(f"Training {training_id} failed")
        db.update_model_status(
            training_id, 
            status="failed", 
            error=str(e)
        )

def start_training(symbol: str, algorithm: str, target_col: str = "close", params: dict = None, data_options: str = None):
    if params is None:
        params = {}
        
    training_id = str(uuid.uuid4())
    
    # Init Record
    db.create_model_record({
        "id": training_id,
        "name": f"{symbol}-{algorithm}-{datetime.now().strftime('%Y%m%d%H%M')}",
        "algorithm": algorithm,
        "symbol": symbol,
        "target_col": target_col,
        "feature_cols": json.dumps([]), # Filled later?
        "hyperparameters": json.dumps(params),
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "metrics": json.dumps({}),
        "data_options": data_options
    })
    
    # Run synchronously for now (or convert to background task if using FastAPI BackgroundTasks)
    # We will call this via BackgroundTasks in main.py
    return training_id
