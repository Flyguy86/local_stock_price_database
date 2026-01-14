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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
try:
    import shap
except ImportError:
    shap = None

from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, r2_score
import joblib
import json
import logging
from datetime import datetime, timezone
import uuid

from .config import settings
from .sync_db_wrapper import db
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


def _save_all_grid_models(grid_search, base_model, X_train, y_train, X_test, y_test, feature_cols_used,
                          symbol, algorithm, target_col, target_transform, timeframe, parent_model_id, db, settings):
    """
    Helper function to save ALL models from a GridSearchCV run.
    Each hyperparameter combination is saved as a separate model in the database.
    """
    all_params = grid_search.cv_results_['params']
    all_scores = grid_search.cv_results_['mean_test_score']
    
    from sklearn.base import clone
    
    for idx, param_set in enumerate(all_params):
        log.info(f"Refitting model {idx+1}/{len(all_params)} with params: {param_set}")
        
        # Clone the pipeline and set hyperparameters
        individual_model = clone(base_model)
        
        # Set the hyperparameters on the cloned pipeline
        for param_name, param_value in param_set.items():
            individual_model.set_params(**{param_name: param_value})
        
        # Fit the model
        individual_model.fit(X_train, y_train)
        
        # Evaluate on test set
        test_preds = individual_model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)
        
        # Save this model to database as separate record
        grid_model_id = str(uuid.uuid4())
        grid_model_path = str(settings.models_dir / f"{grid_model_id}.joblib")
        
        joblib.dump(individual_model, grid_model_path)
        log.info(f"Saved grid model {idx+1} to {grid_model_path}")
        
        # Create database record for this grid model
        grid_record = {
            "id": grid_model_id,
            "symbol": symbol,
            "algorithm": algorithm,
            "target_col": target_col,
            "hyperparameters": param_set,  # Specific params for this model
            "target_transform": target_transform,
            "timeframe": timeframe,
            "feature_cols": feature_cols_used,
            "parent_model_id": parent_model_id,
            "status": "completed",
            "metrics": json.dumps({
                "cv_score": float(all_scores[idx]),
                "test_mse": float(test_mse),
                "test_mae": float(test_mae),
                "test_r2": float(test_r2),
                "grid_search_rank": idx + 1
            }),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_grid_member": True,  # Flag to identify grid search models
            "artifact_path": grid_model_path,
            "name": f"{symbol}-{algorithm}-grid{idx+1}-{datetime.now().strftime('%Y%m%d%H%M')}"
        }
        
        # Insert into database using correct method name
        db.create_model_record(grid_record)
        log.info(f"Inserted grid model {grid_model_id} into database (rank {idx+1}, CV score: {all_scores[idx]:.6f})")
    
    log.info(f"Saved all {len(all_params)} grid search models successfully")


def train_model_task(
    training_id: str, 
    symbol: str, 
    algorithm: str, 
    target_col: str, 
    params: dict, 
    data_options: str = None, 
    timeframe: str = "1m", 
    parent_model_id: str = None, 
    feature_whitelist: list[str] = None, 
    group_id: str = None, 
    target_transform: str = "none", 
    # ElasticNet grids
    alpha_grid: list[float] = None, 
    l1_ratio_grid: list[float] = None,
    # XGBoost grids
    max_depth_grid: list[int] = None,
    min_child_weight_grid: list[int] = None,
    reg_alpha_grid: list[float] = None,  # L1 regularization
    reg_lambda_grid: list[float] = None,  # L2 regularization
    learning_rate_grid: list[float] = None,
    # LightGBM grids
    num_leaves_grid: list[int] = None,
    min_data_in_leaf_grid: list[int] = None,
    lambda_l1_grid: list[float] = None,  # L1 regularization
    lambda_l2_grid: list[float] = None,  # L2 regularization
    lgbm_learning_rate_grid: list[float] = None,
    # RandomForest grids
    rf_max_depth_grid: list = None,
    min_samples_split_grid: list[int] = None,
    min_samples_leaf_grid: list[int] = None,
    n_estimators_grid: list[int] = None,
    max_features_grid: list = None,  # Feature sampling for regularization
    # Control flag
    save_all_grid_models: bool = False
):
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
                # Load parent model features via sync wrapper
                parent_model = db.get_model(parent_model_id)
                if parent_model and parent_model.get('feature_cols'):
                    pf = parent_model['feature_cols']
                    if isinstance(pf, str):
                        parent_features = json.loads(pf)
                    else:
                        parent_features = pf
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
        
        # Parse reference_symbols from data_options JSON if present
        reference_symbols = []
        if data_options:
            try:
                opts_dict = json.loads(data_options)
                reference_symbols = opts_dict.get("reference_symbols", [])
                if reference_symbols:
                    log.info(f"Parsed reference_symbols from data_options: {reference_symbols}")
            except Exception as e:
                log.warning(f"Failed to parse data_options JSON: {e}")
        
        # Build multi-ticker symbol string (primary + references)
        full_symbol = symbol
        if reference_symbols:
            full_symbol = f"{symbol}," + ",".join(reference_symbols)
            log.info(f"Training with multi-ticker TS-aligned data: {full_symbol}")
        
        # 1. Load Data
        df = load_training_data(full_symbol, target_col=target_col, options_filter=data_options, timeframe=timeframe, target_transform=target_transform)
        
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
        
        # Track initial column counts
        columns_initial = len(initial_feature_cols)
        columns_remaining = len(feature_cols_used)
        log.info(f"Column counts after preprocessing: {columns_remaining}/{columns_initial} features (dropped {len(dropped_feature_cols)})")
        
        # Update DB with column counts early so orchestrator can display them
        db.update_model_status(
            training_id,
            status="preprocessing",
            columns_initial=columns_initial,
            columns_remaining=columns_remaining
        )
        
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
        # Conditional Preprocessing logic for Regimes & Heterogeneous Scaling
        
        # 1. Identify Algorithm Category
        is_linear = any(k in algorithm for k in ["linear", "elasticnet", "logistic", "ridge", "lasso", "svm"])
        
        # 2. Categorize Features for Scaling
        # Heuristic Groups:
        # A. Regimes (OneHot) - start with 'regime_vix', 'regime_gmm'
        # B. Robust (Volume/Counts) - 'volume', 'count'
        # C. Bounded (Passthrough/MinMax) - 'rsi', 'ibs', 'aroon', 'stoch', 'bop'
        # D. Standard (Returns, Z-Scores) - everything else
        
        cols_regime = [c for c in X_train.columns if c in ["regime_vix", "regime_gmm"]]
        
        # For Trees, we treat Regimes as Ordinal (do not OneHot). For Linear, we OneHot.
        # But we still need to separate them from scaling.
        
        cols_robust = []
        cols_passthrough = []
        cols_standard = []
        
        remaining_cols = [c for c in X_train.columns if c not in cols_regime]
        
        for c in remaining_cols:
            cl = c.lower()
            if "volume" in cl or "count" in cl or "pro_vol" in cl:
                cols_robust.append(c)
            elif any(x in cl for x in ["rsi", "ibs", "aroon", "stoch", "bop", "mfi", "willr", "ultosc"]):
                # Bounded oscillators: Leave them alone (they preserve their own scale 0-100 or 0-1)
                cols_passthrough.append(c)
            else:
                # Default: Log returns, z-scores, ma_dist, etc.
                cols_standard.append(c)
                
        log.info(f"Scaling Groups determined:")
        log.info(f"  Regimes ({len(cols_regime)}): {cols_regime}")
        log.info(f"  Robust ({len(cols_robust)}): {cols_robust}")
        log.info(f"  Passthrough ({len(cols_passthrough)}): {cols_passthrough}")
        log.info(f"  Standard ({len(cols_standard)}): {len(cols_standard)} cols")

        transformers = []
        
        # 1. Regimes
        if cols_regime:
            if is_linear:
                # OneHot for Linear
                transformers.append(('regime_ohe', OneHotEncoder(handle_unknown='ignore'), cols_regime))
            else:
                # Passthrough (Ordinal) for Trees
                # (Or maybe OrdinalEncoder if they were strings, but they are ints)
                transformers.append(('regime_pass', 'passthrough', cols_regime))
        
        # 2. Robust
        if cols_robust:
            transformers.append(('robust', RobustScaler(), cols_robust))
            
        # 3. Standard
        if cols_standard:
            transformers.append(('standard', StandardScaler(), cols_standard))
            
        # 4. Passthrough (Explicitly listed)
        if cols_passthrough:
            transformers.append(('bounded', 'passthrough', cols_passthrough))
            
        # Create ColumnTransformer
        # remainder='drop' by default, but we covered all columns via loops.
        # To be safe against logic gaps, use remainder='passthrough' with a warning/scaler?
        # Actually proper categorization covers everything. using remainder='passthrough' 
        # is safe to catch anything missed (e.g. regime_sma_dist went to standard? yes).
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough' 
        )

        # Add n_jobs=-1 to tree-based models to utilize all CPUs
        # This makes each individual training job use all available cores
        if algorithm in ['random_forest_regressor', 'random_forest_classifier', 
                         'xgboost_regressor', 'xgboost_classifier',
                         'lightgbm_regressor', 'lightgbm_classifier']:
            if 'n_jobs' not in params and 'n_jobs' not in str(params):
                # Set n_jobs=-1 if not already specified
                params = params.copy() if params else {}
                params['n_jobs'] = -1
                log.info(f"Set n_jobs=-1 for {algorithm} to use all CPU cores")
        
        final_estimator = ModelClass(**params)
        
        # Fix for ElasticNet: sklearn default alpha=1.0 is too high and causes zero coefficients
        # If no params provided and no grid search, use sensible defaults
        if algorithm == "elasticnet_regression" and not params and not (alpha_grid or l1_ratio_grid):
            log.warning("ElasticNet with no params and no grid search - using low-regularization defaults to avoid zero coefficients")
            final_estimator = ModelClass(alpha=0.01, l1_ratio=0.5, max_iter=2000)
        
        model = Pipeline([
            ('preprocessor', preprocessor), 
            ('model', final_estimator)
        ])
        
        tuned_params = None
        
        # --- GRID SEARCH FOR ELASTICNET ---
        # Only do grid search if we have grids provided
        if algorithm == "elasticnet_regression" and (alpha_grid or l1_ratio_grid):
            log.info("Starting Grid Search for ElasticNet (Alpha/L1 Ratio) to avoid zero-feature models...")
            
            # Param Grid - use provided values or defaults
            # Alpha (L2 penalty): higher = more regularization
            # L1 ratio: 0 = pure L2 (Ridge), 1 = pure L1 (Lasso), 0.5 = balanced
            # CRITICAL: Use LOW alphas to avoid over-regularization that zeros coefficients
            default_alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
            default_l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99]
            
            grid_params = {
                'model__alpha': alpha_grid if alpha_grid else default_alphas,
                'model__l1_ratio': l1_ratio_grid if l1_ratio_grid else default_l1_ratios
            }
            log.info(f"Grid: alpha={grid_params['model__alpha']}, l1_ratio={grid_params['model__l1_ratio']}")
            log.info(f"Grid size: {len(grid_params['model__alpha'])} alphas × {len(grid_params['model__l1_ratio'])} l1_ratios = {len(grid_params['model__alpha']) * len(grid_params['model__l1_ratio'])} combinations")

            
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
            log.info(f"Target (y_train) stats: mean={y_train.mean():.6f}, std={y_train.std():.6f}, min={y_train.min():.6f}, max={y_train.max():.6f}")
            log.info(f"Features (X_train) shape: {X_train.shape}")
            
            # DIAGNOSTIC: Check for data issues that could cause all-zero coefficients
            log.info("Checking data quality before training...")
            # 1. Check if target has variance
            if y_train.std() < 1e-10:
                log.error(f"Target has near-zero variance (std={y_train.std():.10f})! ElasticNet will produce all-zero coefficients.")
            # 2. Check if features have variance
            feature_stds = X_train.std()
            zero_var_features = feature_stds[feature_stds < 1e-10].index.tolist() if hasattr(X_train, 'std') else []
            if zero_var_features:
                log.warning(f"Features with near-zero variance: {zero_var_features}")
            # 3. Check correlations with target
            if hasattr(X_train, 'corrwith'):
                correlations = X_train.corrwith(pd.Series(y_train, index=X_train.index))
                max_corr = correlations.abs().max()
                log.info(f"Max absolute correlation with target: {max_corr:.6f} (feature: {correlations.abs().idxmax()})")
                
                # Show top 5 correlated features
                top_corrs = correlations.abs().nlargest(5)
                log.info(f"Top 5 correlated features:")
                for feat, corr in top_corrs.items():
                    log.info(f"  {feat}: {corr:.6f}")
                
                # Data quality warnings
                if max_corr < 0.01:
                    log.error(f"CRITICAL: All features have extremely low correlation with target (max={max_corr:.6f}).")
                    log.error("This indicates NO SIGNAL in the data. Model will likely have all-zero coefficients.")
                    log.error("Check: (1) target alignment, (2) feature engineering, (3) target transform appropriateness.")
                elif max_corr < 0.05:
                    log.warning(f"WARNING: Very weak signal detected (max_corr={max_corr:.6f}).")
                # CRITICAL FIX: Override alpha grid with lower values for weak-signal data
                if max_corr < 0.05:
                    log.warning(f"Weak signal detected (max_corr={max_corr:.6f}). Overriding alpha grid to prevent over-regularization.")
                    original_alphas = grid_params['model__alpha']
                    # Use only alphas <= 0.1, and add smaller ones if needed
                    low_alphas = [a for a in original_alphas if a <= 0.1]
                    if not low_alphas or min(low_alphas) > 0.001:
                        low_alphas = [0.0001, 0.001, 0.01, 0.05, 0.1]
                    grid_params['model__alpha'] = sorted(low_alphas)
                    log.warning(f"Alpha grid reduced from {original_alphas} to {grid_params['model__alpha']} due to weak signal")
            
            grid_search.fit(X_train, y_train)
            
            # --- SAVE ALL GRID MODELS (if requested) ---
            if save_all_grid_models:
                log.info(f"save_all_grid_models=True: Saving ALL {len(grid_search.cv_results_['params'])} models from grid search")
                _save_all_grid_models(grid_search, model, X_train, y_train, X_test, y_test, feature_cols_used,
                                     symbol, algorithm, target_col, target_transform, timeframe, parent_model_id, db, settings)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            log.info(f"Grid Search Best Params: {best_params} | Best Score: {grid_search.best_score_}")
            
            # Check coefficients of best model
            best_estimator = best_model.named_steps['model']
            if hasattr(best_estimator, 'coef_'):
                coefs = best_estimator.coef_
                if coefs.ndim > 1: coefs = coefs[0]
                log.info(f"Best model coefs: shape={coefs.shape}, non-zero={np.count_nonzero(coefs)}, max_abs={np.abs(coefs).max():.6f}, mean_abs={np.abs(coefs).mean():.6f}")
                log.info(f"Best model intercept: {best_estimator.intercept_}")
                
                # FALLBACK: If ElasticNet produces all zeros, try plain LinearRegression (OLS)
                if np.count_nonzero(coefs) == 0:
                    log.warning("ElasticNet grid search produced ALL ZERO coefficients. This indicates VERY WEAK SIGNAL (features have almost no correlation with target).")
                    log.warning("Attempting fallback to LinearRegression (OLS with no regularization) to get baseline...")
                    # LinearRegression already imported at top of file
                    
                    # Create OLS pipeline
                    ols_model = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', LinearRegression())
                    ])
                    ols_model.fit(X_train, y_train)
                    
                    # Check OLS coefficients
                    ols_coefs = ols_model.named_steps['model'].coef_
                    if ols_coefs.ndim > 1: ols_coefs = ols_coefs[0]
                    log.info(f"OLS fallback coefs: shape={ols_coefs.shape}, non-zero={np.count_nonzero(ols_coefs)}, max_abs={np.abs(ols_coefs).max():.6f}, mean_abs={np.abs(ols_coefs).mean():.6f}")
                    
                    if np.count_nonzero(ols_coefs) > 0:
                        log.warning("OLS produced non-zero coefficients. Using OLS instead of ElasticNet.")
                        best_model = ols_model
                        tuned_params = {"algorithm": "LinearRegression (OLS fallback)"}
                    else:
                        log.error("Even OLS (no regularization) produced all-zero coefficients. This strongly indicates the features have NO predictive power for the target.")
                        log.error("Possible causes: (1) target and features misaligned, (2) features are all noise, (3) data leakage was removed, (4) wrong target transform.")
                        tuned_params = best_params
                else:
                    tuned_params = best_params
            else:
                tuned_params = best_params
            
            # Replace model with the tuned one (or OLS fallback)
            model = best_model
        
        # --- GRID SEARCH FOR XGBOOST ---
        elif algorithm in ["xgboost_regressor", "xgboost_classifier"] and XGBRegressor and \
             (max_depth_grid or min_child_weight_grid or reg_alpha_grid or reg_lambda_grid or learning_rate_grid):
            log.info("Starting Grid Search for XGBoost...")
            
            default_max_depth = [3, 4, 5, 6, 7, 8, 9]
            default_min_child_weight = [1, 3, 5, 10, 15, 20, 30]
            default_reg_alpha = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]  # L1
            default_reg_lambda = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]  # L2
            default_learning_rate = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
            
            grid_params = {}
            if max_depth_grid:
                grid_params['model__max_depth'] = max_depth_grid
            else:
                grid_params['model__max_depth'] = default_max_depth
            
            if min_child_weight_grid:
                grid_params['model__min_child_weight'] = min_child_weight_grid
            else:
                grid_params['model__min_child_weight'] = default_min_child_weight
                
            if reg_alpha_grid:
                grid_params['model__reg_alpha'] = reg_alpha_grid
            else:
                grid_params['model__reg_alpha'] = default_reg_alpha
                
            if reg_lambda_grid:
                grid_params['model__reg_lambda'] = reg_lambda_grid
            else:
                grid_params['model__reg_lambda'] = default_reg_lambda
                
            if learning_rate_grid:
                grid_params['model__learning_rate'] = learning_rate_grid
            else:
                grid_params['model__learning_rate'] = default_learning_rate
            
            grid_size = len(grid_params['model__max_depth']) * len(grid_params['model__min_child_weight']) * \
                       len(grid_params['model__reg_alpha']) * len(grid_params['model__reg_lambda']) * len(grid_params['model__learning_rate'])
            log.info(f"XGBoost grid: {len(grid_params['model__max_depth'])} depths × {len(grid_params['model__min_child_weight'])} weights × {len(grid_params['model__reg_alpha'])} alphas(L1) × {len(grid_params['model__reg_lambda'])} lambdas(L2) × {len(grid_params['model__learning_rate'])} LRs = {grid_size} combinations")
            
            grid_search = GridSearchCV(
                model,
                grid_params,
                cv=5,
                scoring='neg_mean_squared_error' if "regressor" in algorithm else 'accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Save all models if requested
            if save_all_grid_models:
                log.info(f"save_all_grid_models=True: Saving ALL {len(grid_search.cv_results_['params'])} XGBoost models from grid search")
                _save_all_grid_models(grid_search, model, X_train, y_train, X_test, y_test, feature_cols_used, 
                                     symbol, algorithm, target_col, target_transform, timeframe, parent_model_id, db, settings)
            
            best_model = grid_search.best_estimator_
            tuned_params = grid_search.best_params_
            log.info(f"XGBoost Grid Search Best Params: {tuned_params} | Best Score: {grid_search.best_score_}")
            model = best_model
        
        # --- GRID SEARCH FOR LIGHTGBM ---
        elif algorithm in ["lightgbm_regressor", "lightgbm_classifier"] and LGBMRegressor and \
             (num_leaves_grid or min_data_in_leaf_grid or lambda_l1_grid or lambda_l2_grid or lgbm_learning_rate_grid):
            log.info("Starting Grid Search for LightGBM...")
            
            default_num_leaves = [7, 15, 31, 63, 95, 127, 191]
            default_min_data_in_leaf = [5, 10, 20, 40, 60, 80, 100]
            default_lambda_l1 = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]  # L1
            default_lambda_l2 = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]  # L2
            default_learning_rate = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
            
            grid_params = {}
            if num_leaves_grid:
                grid_params['model__num_leaves'] = num_leaves_grid
            else:
                grid_params['model__num_leaves'] = default_num_leaves
            
            if min_data_in_leaf_grid:
                grid_params['model__min_data_in_leaf'] = min_data_in_leaf_grid
            else:
                grid_params['model__min_data_in_leaf'] = default_min_data_in_leaf
                
            if lambda_l1_grid:
                grid_params['model__lambda_l1'] = lambda_l1_grid
            else:
                grid_params['model__lambda_l1'] = default_lambda_l1
                
            if lambda_l2_grid:
                grid_params['model__lambda_l2'] = lambda_l2_grid
            else:
                grid_params['model__lambda_l2'] = default_lambda_l2
                
            if lgbm_learning_rate_grid:
                grid_params['model__learning_rate'] = lgbm_learning_rate_grid
            else:
                grid_params['model__learning_rate'] = default_learning_rate
            
            grid_size = len(grid_params['model__num_leaves']) * len(grid_params['model__min_data_in_leaf']) * \
                       len(grid_params['model__lambda_l1']) * len(grid_params['model__lambda_l2']) * len(grid_params['model__learning_rate'])
            log.info(f"LightGBM grid: {len(grid_params['model__num_leaves'])} leaves × {len(grid_params['model__min_data_in_leaf'])} min_data × {len(grid_params['model__lambda_l1'])} l1 × {len(grid_params['model__lambda_l2'])} l2 × {len(grid_params['model__learning_rate'])} LRs = {grid_size} combinations")
            
            grid_search = GridSearchCV(
                model,
                grid_params,
                cv=5,
                scoring='neg_mean_squared_error' if "regressor" in algorithm else 'accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Save all models if requested
            if save_all_grid_models:
                log.info(f"save_all_grid_models=True: Saving ALL {len(grid_search.cv_results_['params'])} LightGBM models from grid search")
                _save_all_grid_models(grid_search, model, X_train, y_train, X_test, y_test, feature_cols_used,
                                     symbol, algorithm, target_col, target_transform, timeframe, parent_model_id, db, settings)
            
            best_model = grid_search.best_estimator_
            tuned_params = grid_search.best_params_
            log.info(f"LightGBM Grid Search Best Params: {tuned_params} | Best Score: {grid_search.best_score_}")
            model = best_model
        
        # --- GRID SEARCH FOR RANDOMFOREST ---
        elif algorithm in ["random_forest_regressor", "random_forest_classifier"] and \
             (rf_max_depth_grid or min_samples_split_grid or min_samples_leaf_grid or n_estimators_grid or max_features_grid):
            log.info("Starting Grid Search for RandomForest...")
            
            default_max_depth = [5, 10, 15, 20, 30, 50, None]
            default_min_samples_split = [2, 5, 10, 20, 30, 50, 100]
            default_min_samples_leaf = [1, 2, 4, 8, 12, 16, 20]
            default_n_estimators = [25, 50, 75, 100, 150, 200, 300]
            default_max_features = ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9, 1.0]  # Feature sampling
            
            grid_params = {}
            if rf_max_depth_grid is not None:
                grid_params['model__max_depth'] = rf_max_depth_grid
            else:
                grid_params['model__max_depth'] = default_max_depth
            
            if min_samples_split_grid:
                grid_params['model__min_samples_split'] = min_samples_split_grid
            else:
                grid_params['model__min_samples_split'] = default_min_samples_split
                
            if min_samples_leaf_grid:
                grid_params['model__min_samples_leaf'] = min_samples_leaf_grid
            else:
                grid_params['model__min_samples_leaf'] = default_min_samples_leaf
                
            if n_estimators_grid:
                grid_params['model__n_estimators'] = n_estimators_grid
            else:
                grid_params['model__n_estimators'] = default_n_estimators
                
            if max_features_grid:
                grid_params['model__max_features'] = max_features_grid
            else:
                grid_params['model__max_features'] = default_max_features
            
            grid_size = len(grid_params['model__max_depth']) * len(grid_params['model__min_samples_split']) * \
                       len(grid_params['model__min_samples_leaf']) * len(grid_params['model__n_estimators']) * len(grid_params['model__max_features'])
            log.info(f"RandomForest grid: {len(grid_params['model__max_depth'])} depths × {len(grid_params['model__min_samples_split'])} splits × {len(grid_params['model__min_samples_leaf'])} leafs × {len(grid_params['model__n_estimators'])} estimators × {len(grid_params['model__max_features'])} max_features = {grid_size} combinations")
            
            grid_search = GridSearchCV(
                model,
                grid_params,
                cv=5,
                scoring='neg_mean_squared_error' if "regressor" in algorithm else 'accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Save all models if requested
            if save_all_grid_models:
                log.info(f"save_all_grid_models=True: Saving ALL {len(grid_search.cv_results_['params'])} RandomForest models from grid search")
                _save_all_grid_models(grid_search, model, X_train, y_train, X_test, y_test, feature_cols_used,
                                     symbol, algorithm, target_col, target_transform, timeframe, parent_model_id, db, settings)
            
            best_model = grid_search.best_estimator_
            tuned_params = grid_search.best_params_
            log.info(f"RandomForest Grid Search Best Params: {tuned_params} | Best Score: {grid_search.best_score_}")
            model = best_model

        else:
            # 4. Train Final Model (Normal)
            model.fit(X_train, y_train)
        
        # 5. Evaluate Final Model (for feature analysis and legacy metric report)
        preds = model.predict(X_test)
        
        metrics = {}
        if tuned_params:
            metrics["tuned_params"] = tuned_params
            
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
            preprocessor = model.named_steps['preprocessor']
            
            if hasattr(estimator, "feature_importances_"):
                # Tree-based model
                imps = estimator.feature_importances_
                # Check if preprocessing expanded features (e.g., OneHotEncoder)
                if hasattr(preprocessor, 'get_feature_names_out'):
                    try:
                        expanded_names = preprocessor.get_feature_names_out()
                        if len(expanded_names) == len(imps):
                            # Map expanded features back to original features by prefix
                            for i, expanded_name in enumerate(expanded_names):
                                # Try to match to original feature (e.g., "regime_ohe__regime_vix_1" -> "regime_vix")
                                for orig_col in feature_cols_used:
                                    if orig_col in expanded_name:
                                        if "tree_importance" not in feature_details[orig_col]:
                                            feature_details[orig_col]["tree_importance"] = 0.0
                                        feature_details[orig_col]["tree_importance"] += float(imps[i])
                                        break
                        else:
                            # Dimension mismatch, use direct mapping if sizes match
                            if len(imps) == len(feature_cols_used):
                                for i, col in enumerate(feature_cols_used):
                                    feature_details[col]["tree_importance"] = float(imps[i])
                    except:
                        # Fallback to direct mapping
                        if len(imps) == len(feature_cols_used):
                            for i, col in enumerate(feature_cols_used):
                                feature_details[col]["tree_importance"] = float(imps[i])
                else:
                    # No preprocessing or simple preprocessing
                    for i, col in enumerate(feature_cols_used):
                        feature_details[col]["tree_importance"] = float(imps[i])
                        
            elif hasattr(estimator, "coef_"):
                # Linear model
                coefs = estimator.coef_
                if coefs.ndim > 1: 
                    coefs = coefs[0]
                
                log.info(f"Raw coefficient array shape: {coefs.shape}, non-zero: {np.count_nonzero(coefs)}, max abs: {np.abs(coefs).max():.6f}")
                    
                # Check if preprocessing expanded features (e.g., OneHotEncoder)
                if hasattr(preprocessor, 'get_feature_names_out'):
                    try:
                        expanded_names = preprocessor.get_feature_names_out()
                        log.info(f"Expanded feature names count: {len(expanded_names)}, coefs count: {len(coefs)}")
                        
                        # DEBUG: Show first 10 expanded names and their coefficients
                        log.info("First 10 expanded features and coefficients:")
                        for i in range(min(10, len(expanded_names))):
                            log.info(f"  [{i}] {expanded_names[i]}: {coefs[i]:.6f}")
                        
                        if len(expanded_names) == len(coefs):
                            # Map expanded features back to original features by prefix
                            mapped_count = 0
                            unmapped_features = []
                            
                            for i, expanded_name in enumerate(expanded_names):
                                matched = False
                                # Try to match to original feature (e.g., "regime_ohe__regime_vix_1" -> "regime_vix")
                                for orig_col in feature_cols_used:
                                    if orig_col in expanded_name:
                                        if "coefficient" not in feature_details[orig_col]:
                                            feature_details[orig_col]["coefficient"] = 0.0
                                        feature_details[orig_col]["coefficient"] += float(coefs[i])
                                        mapped_count += 1
                                        matched = True
                                        break
                                
                                if not matched and abs(coefs[i]) > 1e-10:
                                    # Only log unmapped features with non-zero coefficients
                                    unmapped_features.append((expanded_name, coefs[i]))
                                    
                            log.info(f"Mapped {mapped_count} coefficients to {len(feature_cols_used)} original features")
                            
                            if unmapped_features:
                                log.warning(f"Found {len(unmapped_features)} unmapped features with non-zero coefficients:")
                                for name, coef in unmapped_features[:5]:  # Show first 5
                                    log.warning(f"  {name}: {coef:.6f}")
                        else:
                            # Dimension mismatch, use direct mapping if sizes match
                            log.warning(f"Expanded names ({len(expanded_names)}) != coefs ({len(coefs)})")
                            if len(coefs) == len(feature_cols_used):
                                for i, col in enumerate(feature_cols_used):
                                    feature_details[col]["coefficient"] = float(coefs[i])
                    except Exception as map_error:
                        log.warning(f"Error mapping expanded features: {map_error}")
                        # Fallback to direct mapping
                        if len(coefs) == len(feature_cols_used):
                            for i, col in enumerate(feature_cols_used):
                                feature_details[col]["coefficient"] = float(coefs[i])
                else:
                    # No preprocessing or simple preprocessing
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
            
            log.info(f"Computing permutation importance on {len(X_perm)} test samples with {X_perm.shape[1]} features")
            
            if len(X_perm) > 10:
                perm_result = permutation_importance(model, X_perm, y_perm, n_repeats=5, random_state=42, scoring=scorer)
                log.info(f"Permutation importance: shape={perm_result.importances_mean.shape}, non-zero={np.count_nonzero(perm_result.importances_mean)}, max={np.abs(perm_result.importances_mean).max():.6f}")
                
                # The permutation importance is for the RAW input features (before preprocessing)
                # so we can map directly to feature_cols_used
                if len(perm_result.importances_mean) == len(feature_cols_used):
                    for i, col in enumerate(feature_cols_used):
                        feature_details[col]["permutation_mean"] = float(perm_result.importances_mean[i])
                else:
                    log.warning(f"Permutation importance shape mismatch: {len(perm_result.importances_mean)} != {len(feature_cols_used)}")
        except Exception as e:
             log.warning(f"Error extracting permutation importance: {e}")

        # 3. SHAP Values
        if shap:
            try:
                estimator = model.named_steps['model']
                preprocessor = model.named_steps['preprocessor']

                # Use a small background sample for explainers
                # SCALE THE DATA first
                X_bg = preprocessor.transform(X_train[:100])
                X_eval = preprocessor.transform(X_test[:100])
                
                # If OHE was used, the number of features in X_bg might be larger than original feature_cols_used
                # This breaks the mapping back to feature names.
                # For now, if we detect ColumnTransformer, we skip SHAP mapping or accept it might fail for OHE cols.
                is_ohe = isinstance(preprocessor, ColumnTransformer)
                
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
                
                if explainer and len(X_eval) > 0 and not is_ohe:
                     # Only run SHAP mapping if dimensions match (no OHE expansion)
                     # Or logic needs to retrieve feature names from transformer
                     shap_vals = explainer.shap_values(X_eval)
                     # For classification, shap_vals might be a list of arrays (one per class). Take the first (or positive class)
                     if isinstance(shap_vals, list):
                         shap_vals = shap_vals[-1] # Positive class
                         
                     # Mean Absolute SHAP value per feature
                     shap_mean = np.mean(np.abs(shap_vals), axis=0)
                     for i, col in enumerate(feature_cols_used):
                         feature_details[col]["shap_mean_abs"] = float(shap_mean[i])
                     
                     # ----------------------------------------------------
                     # 4. Regime Interaction Analysis
                     # ----------------------------------------------------
                     # Analyze how feature importance varies by regime.
                     # Only run when we have valid SHAP values mapped to original features.
                     try:
                         regime_interaction = {}
                         # Valid indices for the SHAP evaluation set
                         eval_indices = X_test.index[:100]
                         if len(eval_indices) > 0:
                             # Check original DF for regime columns
                             regime_cols = [c for c in df.columns if str(c).startswith("regime_")]
                             
                             for r_col in regime_cols:
                                 # Get regime labels for the evaluation set
                                 # Ensure alignment
                                 r_vals = df.loc[eval_indices, r_col]
                                 unique_regimes = r_vals.unique()
                                 
                                 # Only analyze meaningful regimes (discrete categories, not continuous)
                                 if len(unique_regimes) < 10: 
                                     regime_interaction[r_col] = {}
                                     
                                     for r_val in unique_regimes:
                                         # Create a mask for this regime
                                         mask = (r_vals == r_val).values
                                         if mask.sum() > 0:
                                             # Filter SHAP values for this regime
                                             # shap_vals shape: (samples, features)
                                             r_shap_vals = shap_vals[mask]
                                             
                                             # Calculate Mean Abs SHAP for this regime
                                             r_shap_mean = np.mean(np.abs(r_shap_vals), axis=0)
                                             
                                             # Store raw values
                                             # Convert numpy types to native Python for JSON
                                             r_dict = {}
                                             for i, col in enumerate(feature_cols_used):
                                                 r_dict[col] = float(r_shap_mean[i])
                                             
                                             r_key = str(r_val) # transform 0/1 to "0"/"1" for JSON
                                             regime_interaction[r_col][r_key] = r_dict
                                             
                         if regime_interaction:
                             metrics["regime_importance"] = regime_interaction
                             log.info(f"Calculated Regime-Conditional Feature Importance for: {list(regime_interaction.keys())}")
                             
                     except Exception as reg_err:
                         log.warning(f"Failed to calc regime interactions: {reg_err}")
                     # ----------------------------------------------------
                         
                elif explainer and is_ohe:
                     log.info("Skipping detailed SHAP mapping due to OHE expansion (ColumnTransformer).")

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
