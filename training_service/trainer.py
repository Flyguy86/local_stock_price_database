import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
try:
    from xgboost import XGBRegressor, XGBClassifier
except (ImportError, OSError):
    XGBRegressor = None
    XGBClassifier = None
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except (ImportError, OSError):
    LGBMRegressor = None
    LGBMClassifier = None
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
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
    "random_forest_classifier": RandomForestClassifier,
    "elasticnet_regression": ElasticNet
}
if XGBRegressor:
    ALGORITHMS["xgboost_regressor"] = XGBRegressor
    ALGORITHMS["xgboost_classifier"] = XGBClassifier
if LGBMRegressor:
    ALGORITHMS["lightgbm_regressor"] = LGBMRegressor
    ALGORITHMS["lightgbm_classifier"] = LGBMClassifier
    ALGORITHMS["lightgbm_classifier"] = LGBMClassifier

def train_model_task(training_id: str, symbol: str, algorithm: str, target_col: str, params: dict, data_options: str = None, timeframe: str = "1m", parent_model_id: str = None, feature_whitelist: list[str] = None, group_id: str = None, target_transform: str = "none"):
    model_path = str(settings.models_dir / f"{training_id}.joblib")
    
    try:
        log.info(f"Starting training {training_id} for {symbol} using {algorithm} at {timeframe}. TargetTransform: {target_transform}")

        # If group_id provided, simulate a shared "Name" for the UI if needed, 
        # but the db record will store the specific target.

        # Load Parent Features if specified (unless explicit whitelist provided)
        parent_features = None
        if feature_whitelist:
             # Explicit whitelist from UI overrides parent lookup (or is derived from it)
             parent_features = feature_whitelist
             log.info(f"Using {len(feature_whitelist)} whitelisted features from request")
        elif parent_model_id:
            try:
                # Fallback: keep all parent features if not explicitly selected
                with db.get_connection() as conn:
                    pf = conn.execute("SELECT feature_cols FROM models WHERE id = ?", [parent_model_id]).fetchone()
                    if pf and pf[0]:
                        parent_features = json.loads(pf[0])
                        log.info(f"Loaded {len(parent_features)} features from parent model {parent_model_id}")
            except Exception as e:
                log.warning(f"Failed to load parent model features: {e}")

        # Extract pruning setting from params (default 0.05)
        p_val_thresh = 0.05
        if params and "p_value_threshold" in params:
            try:
                p_val_thresh = float(params.pop("p_value_threshold"))
            except:
                p_val_thresh = 0.05
        
        # 1. Load Data
        df = load_training_data(symbol, target_col=target_col, options_filter=data_options, timeframe=timeframe, target_transform=target_transform)
        
        # Set TS as index for explicit alignment
        if "ts" in df.columns:
            df = df.set_index("ts").sort_index()
            log.info("Set 'ts' as DataFrame index for alignment verification.")

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
        # Drop metadata columns and target AND the raw target_col (which we kept for ref but is leakage if used as feature)
        drop_cols = ["target", "ts", "symbol", "date", "source", "options", "target_col_shifted", target_col]
        if split_col_name:
            drop_cols.append(split_col_name)
            
        # Filter for numeric columns only to be safe
        df_numeric = df.select_dtypes(include=[np.number])
        feature_cols = [c for c in df_numeric.columns if c not in drop_cols]
        
        if parent_features:
            # INTERSECT: Only keep features that were present in parent AND are in current dataset
            original_count = len(feature_cols)
            feature_cols = [c for c in feature_cols if c in parent_features]
            log.info(f"Applied parent feature mask: {original_count} -> {len(feature_cols)} features")

        if not feature_cols:
            raise ValueError("No numeric feature columns found for training (check parent model intersection?)")

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
            
        # DEBUG: Log data shape and types
        log.info(f"Data Loaded: X.shape={X.shape}, y.shape={y.shape}")
        log.info(f"Feature Types: {X.dtypes.value_counts().to_dict()}")
        if len(X) < 10:
             log.warning("Data Set extremely small (<10 rows).")

        feature_cols_used = list(X.columns)
        dropped_feature_cols = sorted(list(set(initial_feature_cols) - set(feature_cols_used)))
        
        X = X[feature_cols_used] # Update X to only valid cols

        # Ensure we only have numeric features
        # 1m data might contain object columns (strings) that break sklearn
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < len(feature_cols_used):
             dropped_non_numeric = list(set(feature_cols_used) - set(numeric_cols))
             log.warning(f"Dropping {len(dropped_non_numeric)} non-numeric features: {dropped_non_numeric}")
             feature_cols_used = numeric_cols
             X = X[feature_cols_used]
             
        if X.empty:
            raise ValueError("No numeric features available for training.")

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

        # --- Top Down Pruning Step ---
        # Filter by P-value (Univariate F-test to avoid multicollinearity checks for now)
        if p_val_thresh < 1.0 and len(feature_cols_used) > 0:
            try:
                log.info(f"Pruning features with P-value > {p_val_thresh}")
                
                # Prepare data for selection (Impute + Standardize)
                X_sel = X.copy()
                if X_sel.isna().any().any():
                     imp_sel = SimpleImputer(strategy='mean')
                     X_sel_arr = imp_sel.fit_transform(X_sel)
                     X_sel = pd.DataFrame(X_sel_arr, columns=X_sel.columns, index=X_sel.index)

                log.info(f"Data ready for pruning check. X shape: {X_sel.shape}, y shape: {y.shape}")
                log.info(f"X types sample: {X_sel.dtypes.value_counts()}")

                # Standardize (User req: "Standardize your features")
                scaler_sel = StandardScaler()
                X_sel_scaled = pd.DataFrame(scaler_sel.fit_transform(X_sel), columns=X_sel.columns)

                # Calculate P-values
                # f_regression returns (F, p_values)
                # It is robust for initial pruning.
                if X_sel_scaled.shape[1] == 0:
                     raise ValueError("X has 0 features after preprocessing, cannot run f_regression")
                if len(y) == 0:
                     raise ValueError("y has 0 samples, cannot run f_regression")

                _, p_vals = f_regression(X_sel_scaled, y)
                
                # Filter
                keep_mask = p_vals < p_val_thresh
                kept_feats = [f for f, k in zip(feature_cols_used, keep_mask) if k]
                removed_feats = sorted(list(set(feature_cols_used) - set(kept_feats)))
                
                if removed_feats:
                    log.info(f"Dropped {len(removed_feats)} features: {removed_feats}")
                    feature_cols_used = kept_feats
                    X = X[feature_cols_used]
                    
                    if not feature_cols_used:
                         log.warning("All features pruned! Reverting to top 1 feature.")
                         # Fallback: keep lowest p-value
                         best_idx = np.argmin(p_vals)
                         feature_cols_used = [X_sel.columns[best_idx]]
                         X = X.iloc[:, [best_idx]]

                # Log Standardized Coefficients (Ranking)
                if len(feature_cols_used) > 0:
                    lr_an = LinearRegression()
                    lr_an.fit(X_sel_scaled[feature_cols_used], y)
                    # Get Coefs
                    coefs = pd.Series(lr_an.coef_, index=feature_cols_used)
                    # Sort by Abs
                    ranked = coefs.abs().sort_values(ascending=False)
                    log.info(f"Feature Ranking (Std Beta): {ranked.head(10).to_dict()}")

            except Exception as e:
                import traceback
                log.error(f"Pruning failed: {e}. Trace: {traceback.format_exc()}. Proceeding with all features.")

        # Split Data (Custom Column or Time-based)
        is_cv = False
        cv_metrics = []
        X_train_final, y_train_final = None, None
        X_test_final, y_test_final = None, None
        
        if split_col_name:
            log.info(f"Splitting data using column: {split_col_name}")
            # Aligned split values
            split_vals = df.loc[X.index, split_col_name].astype(str).str.lower()
            
            # Detect Blocks for Potential CV
            # Create a group ID that increments every time the value changes (e.g. train -> test -> train)
            # This identifies contiguous chunks
            block_ids = (split_vals != split_vals.shift()).cumsum()
            blocks = split_vals.groupby(block_ids)
            
            unique_blocks = list(blocks.groups.keys())
            
            # If we have repeated train/test patterns (e.g., T, T, T, t, t, T, T...), we treat it as CV
            # We look for (Train -> Test) pairs.
            folds = []
            current_train_idx = None
            
            # Iterate through blocks to find T -> t sequences
            for bid in unique_blocks:
                idx = blocks.groups[bid]
                val = split_vals.loc[idx[0]]
                if val == 'train':
                    current_train_idx = idx
                elif val == 'test' and current_train_idx is not None:
                    folds.append((current_train_idx, idx))
                    current_train_idx = None # Consume the train block
            
            if len(folds) > 1:
                log.info(f"Detected {len(folds)} folds. enabling Cross-Validation loop.")
                is_cv = True
                
                # CV Loop
                for i, (tr_idx, te_idx) in enumerate(folds):
                    f_X_train, f_X_test = X.loc[tr_idx], X.loc[te_idx]
                    f_y_train, f_y_test = y.loc[tr_idx], y.loc[te_idx]
                    
                    # Impute
                    if f_X_train.isna().any().any() or f_X_test.isna().any().any():
                         imp = SimpleImputer(strategy='mean')
                         f_X_train = imp.fit_transform(f_X_train)
                         f_X_test = imp.transform(f_X_test)

                    # Train Fold
                    ModelClass = ALGORITHMS.get(algorithm)
                    f_model = ModelClass(**params)
                    f_model.fit(f_X_train, f_y_train)
                    
                    # Eval Fold
                    f_preds = f_model.predict(f_X_test)
                    
                    f_met = {}
                    if "regressor" in algorithm or "regression" in algorithm:
                        f_mse = mean_squared_error(f_y_test, f_preds)
                        f_met["mse"] = f_mse
                        f_met["rmse"] = float(f_mse ** 0.5)
                    else:
                        f_acc = accuracy_score(f_y_test, f_preds)
                        f_met["accuracy"] = f_acc
                    cv_metrics.append(f_met)
                    log.info(f"Fold {i+1} metrics: {f_met}")

                # Prepare Final Dataset: Use ALL train/test rows found for the final model
                # This ensures we include even orphan train blocks (e.g. latest data without a test set yet)
                train_mask = split_vals == 'train'
                test_mask = split_vals == 'test'
                
                X_train = X[train_mask]
                X_test = X[test_mask]
                y_train = y[train_mask]
                y_test = y[test_mask]
                
            else:
                # Standard Single Split logic (if only 1 Pair found, or scrambled data)
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
        
        # 3. Impute Missing Values in Features (Mean Strategy) - Final Model
        # This fills gaps (e.g. from indicators) without dropping data
        if X_train.isna().any().any() or X_test.isna().any().any():
            log.info("Imputing missing feature values with mean (Final Model)")
            imputer = SimpleImputer(strategy='mean')
            # Fit on train, transform both
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            # Note: X_train/X_test are now numpy arrays, which SKLearn accepts fine
            
        # 3. Initialize Final Model
        ModelClass = ALGORITHMS.get(algorithm)
        if not ModelClass:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        if "regressor" in algorithm or "regression" in algorithm:
             if target_transform in ["log_return", "pct_change"]:
                  pass # Correct matching
        elif target_transform in ["log_return", "pct_change"]:
             log.warning("Classification algorithm selected but regression transform used. Target remains continuous. Did you mean to use Regressor?")

        # User Req: "Standardize your features" (Pipeline)
        # We wrap the estimator in a pipeline.
        final_estimator = ModelClass(**params)
        model = Pipeline([
            ('scaler', StandardScaler()), 
            ('model', final_estimator)
        ])
        
        # --- GRID SEARCH FOR ELASTICNET ---
        if algorithm == "elasticnet_regression" and not params:
            log.info("Starting Grid Search for ElasticNet (Alpha/L1 Ratio) to avoid zero-feature models...")
            
            # Param Grid
            # alpha: Regularization strength. Lower = Less pruning.
            # l1_ratio: Mix of Lasso (L1) / Ridge (L2). 
            # 1.0 = Lasso (high sparsity, features -> 0)
            # 0.0 = Ridge (low sparsity, features -> small weights)
            # We want to find a balance where features don't all hit zero.
            grid_params = {
                'model__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
                'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
            }
            
            # Use GridSearchCV
            # n_jobs=-1 uses all CPU cores (Multi-threaded)
            # cv=3 for speed given time-series constraints (simple k-fold here, ideally TimeSeriesSplit but keeping it simple for now)
            grid_search = GridSearchCV(
                model, 
                grid_params, 
                cv=5, 
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            log.info(f"Fitting Grid Search on {len(X_train)} samples...")
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            log.info(f"Grid Search Best Params: {best_params} | Best Score: {grid_search.best_score_}")
            
            # Replace model with the tuned one
            model = best_model
            
            # Capture selected params into metrics for reporting
            metrics["tuned_params"] = best_params

        else:
            # 4. Train Final Model (Normal)
            model.fit(X_train, y_train)
        
        # 5. Evaluate Final Model (for feature analysis and legacy metric report)
        preds = model.predict(X_test)
        
        metrics = {}
        if dropped_feature_cols:
            metrics["dropped_cols"] = dropped_feature_cols
        metrics["features_count"] = len(feature_cols_used)
        
        metrics["target_transform"] = target_transform

        if is_cv:
            # Process CV metrics
            if "accuracy" in cv_metrics[0]:
                mean_acc = sum(m["accuracy"] for m in cv_metrics) / len(cv_metrics)
                metrics["accuracy"] = mean_acc
                metrics["cv_folds"] = len(cv_metrics)
                metrics["cv_detail"] = cv_metrics
            if "mse" in cv_metrics[0]:
                mean_mse = sum(m["mse"] for m in cv_metrics) / len(cv_metrics)
                metrics["mse"] = mean_mse
                metrics["rmse"] = float(mean_mse ** 0.5)
                metrics["cv_folds"] = len(cv_metrics)
        else:
            if "regressor" in algorithm or "regression" in algorithm:
                mse = mean_squared_error(y_test, preds)
                metrics["mse"] = mse
                metrics["rmse"] = float(mse ** 0.5)
                
                # --- PRICE RECONSTRUCTION METRIC ---
                # User wants to verify RMSE in Price units ($) even if we trained on Log Returns
                try:
                    if target_transform in ["log_return", "pct_change"] and target_col in df.columns:
                        # 1. Get Base Prices for Test Set
                        # Note: X_test was sliced sequentially or via mask.
                        # We need the matching rows from df.
                        # X_test index should match df index if we preserved it.
                        test_indices = X_test.index
                        base_prices = df.loc[test_indices, target_col]
                        
                        if target_transform == "log_return":
                            # Pred Price = Base * exp(Pred Log Ret)
                            # True Price = Base * exp(True Log Ret)
                            # (Or just use Future Price if we had it? We can derive True Price)
                            rec_preds = base_prices * np.exp(preds)
                            rec_true = base_prices * np.exp(y_test)
                        elif target_transform == "pct_change":
                            rec_preds = base_prices * (1 + preds)
                            rec_true = base_prices * (1 + y_test)
                            
                        rec_mse = mean_squared_error(rec_true, rec_preds)
                        metrics["rmse_price"] = float(rec_mse ** 0.5)
                        metrics["rmse_price_unit"] = "$"
                        log.info(f"Reconstructed Price RMSE: {metrics['rmse_price']}")
                except Exception as rec_err:
                    log.warning(f"Failed to calculate reconstructed price metrics: {rec_err}")
                # -----------------------------------

            else:
                acc = accuracy_score(y_test, preds)
                metrics["accuracy"] = acc

        # Initialize detailed feature info
        feature_details = {col: {} for col in feature_cols_used}

        # 1. Linear Coefficients / Tree Importance
        try:
            # Unwrap pipeline
            estimator = model.named_steps['model']
            
            if hasattr(estimator, "feature_importances_"):
                imps = estimator.feature_importances_
                for i, col in enumerate(feature_cols_used):
                    feature_details[col]["tree_importance"] = float(imps[i])
            elif hasattr(estimator, "coef_"):
                coefs = estimator.coef_
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
                estimator = model.named_steps['model']
                scaler = model.named_steps['scaler']

                # Use a small background sample for explainers
                # SCALE THE DATA first
                X_bg = scaler.transform(X_train[:100])
                X_eval = scaler.transform(X_test[:100])
                
                explainer = None
                est_type = str(type(estimator))
                
                if "Linear" in est_type or "Logistic" in est_type:
                     explainer = shap.LinearExplainer(estimator, X_bg)
                elif any(k in est_type for k in ["Forest", "Tree", "XGB", "LGBM", "Boosting", "CatBoost"]):
                     try:
                        explainer = shap.TreeExplainer(estimator)
                     except Exception as tree_err:
                        log.warning(f"TreeExplainer failed for {est_type}, falling back to Explainer: {tree_err}")
                        explainer = shap.Explainer(estimator, X_bg)
                else:
                     # Fallback
                     explainer = shap.Explainer(estimator, X_bg)
                
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
            feature_cols=json.dumps(feature_cols_used),
            target_transform=target_transform
        )
        log.info(f"Training {training_id} completed. Metrics: {metrics}")
        
    except Exception as e:
        log.exception(f"Training {training_id} failed")
        db.update_model_status(
            training_id, 
            status="failed", 
            error=str(e)
        )

def start_training(symbol: str, algorithm: str, target_col: str = "close", params: dict = None, data_options: str = None, timeframe: str = "1m", parent_model_id: str = None, group_id: str = None, target_transform: str = "none"):
    if params is None:
        params = {}
        
    training_id = str(uuid.uuid4())
    
    # Init Record
    db.create_model_record({
        "id": training_id,
        "name": f"{symbol}-{algorithm}-{target_col}-{timeframe}-{datetime.now().strftime('%Y%m%d%H%M')}",
        "algorithm": algorithm,
        "symbol": symbol,
        "target_col": target_col,
        "feature_cols": json.dumps([]),
        "hyperparameters": json.dumps(params),
        "status": "pending",
        "metrics": json.dumps({}),
        "data_options": data_options,
        "parent_model_id": parent_model_id,
        "group_id": group_id,
        "timeframe": timeframe,
        "target_transform": target_transform
    })
    
    return training_id
