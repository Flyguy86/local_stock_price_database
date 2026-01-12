"""
Hybrid Stacking Regressor combining Linear Regression with XGBoost.

Architecture:
    Base Model: Linear Regression (learns linear patterns from context symbols)
    Meta Model: XGBoost (learns non-linear patterns from base predictions + volatility/time features)
    
This allows the model to capture both linear market relationships and complex
non-linear patterns in a two-stage learning process.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
import xgboost as xgb
import logging

log = logging.getLogger(__name__)


class HybridRegressor(BaseEstimator, RegressorMixin):
    """
    Stacking Regressor with Linear Regression base and XGBoost meta-learner.
    
    Training Process:
        1. Train base Linear Regression on context features (QQQ, MSFT, GOOGL close prices)
        2. Generate out-of-fold predictions from base model
        3. Train XGBoost meta-model on:
           - Base model predictions
           - ATR (volatility)
           - Time features (time_sin, time_cos)
    
    Prediction Process:
        1. Get base model prediction
        2. Combine with ATR/time features
        3. Pass to meta model for final prediction
    """
    
    def __init__(
        self,
        base_features=None,
        meta_features=None,
        xgb_params=None,
        cv_folds=5
    ):
        """
        Args:
            base_features: List of column names for base model (e.g., context symbols)
            meta_features: List of column names for meta model (e.g., ['atr_14', 'time_sin', 'time_cos'])
            xgb_params: Dict of XGBoost hyperparameters
            cv_folds: Number of cross-validation folds for generating base predictions
        """
        self.base_features = base_features or []
        self.meta_features = meta_features or ['atr_14', 'time_sin', 'time_cos']
        self.xgb_params = xgb_params or {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        self.cv_folds = cv_folds
        
        self.base_model = LinearRegression()
        self.meta_model = xgb.XGBRegressor(**self.xgb_params)
        
        # Track feature names for sklearn compatibility
        self.feature_names_in_ = None
        
    def _extract_features(self, X):
        """Extract base and meta features from input DataFrame or array."""
        if isinstance(X, pd.DataFrame):
            # Auto-detect context symbol features if not specified
            if not self.base_features:
                # Look for columns ending with symbol names (e.g., close_QQQ, close_MSFT)
                context_cols = [c for c in X.columns if any(sym in c for sym in ['QQQ', 'MSFT', 'GOOGL'])]
                self.base_features = context_cols
                log.info(f"Auto-detected base features: {self.base_features}")
            
            X_base = X[self.base_features].values if self.base_features else np.array([]).reshape(len(X), 0)
            X_meta_direct = X[self.meta_features].values if self.meta_features else np.array([]).reshape(len(X), 0)
            
            # Store feature names for later
            if self.feature_names_in_ is None:
                self.feature_names_in_ = list(X.columns)
            
        else:
            # Assume numpy array with all features
            n_base = len(self.base_features)
            X_base = X[:, :n_base] if n_base > 0 else np.array([]).reshape(X.shape[0], 0)
            X_meta_direct = X[:, n_base:] if n_base < X.shape[1] else np.array([]).reshape(X.shape[0], 0)
        
        return X_base, X_meta_direct
    
    def fit(self, X, y):
        """
        Fit the hybrid stacking model.
        
        Steps:
            1. Train base Linear Regression on context features
            2. Generate cross-validated predictions from base model (avoids overfitting)
            3. Combine base predictions with meta features (ATR, time)
            4. Train XGBoost meta-model on combined features
        """
        log.info("Training HybridRegressor (Stacking: LinearRegression → XGBoost)")
        
        X_base, X_meta_direct = self._extract_features(X)
        
        # Step 1: Train base model on context features
        if X_base.shape[1] > 0:
            log.info(f"Training base LinearRegression on {X_base.shape[1]} context features")
            self.base_model.fit(X_base, y)
            
            # Step 2: Generate out-of-fold predictions to avoid overfitting
            log.info(f"Generating cross-validated predictions with {self.cv_folds} folds")
            base_predictions = cross_val_predict(
                self.base_model, 
                X_base, 
                y, 
                cv=self.cv_folds
            ).reshape(-1, 1)
            
            log.info(f"Base model cross-val R² (approximate): {np.corrcoef(base_predictions.flatten(), y)[0,1]**2:.4f}")
        else:
            log.warning("No base features provided, skipping base model")
            base_predictions = np.zeros((len(y), 1))
        
        # Step 3: Combine base predictions with meta features
        if X_meta_direct.shape[1] > 0:
            X_meta = np.hstack([base_predictions, X_meta_direct])
            log.info(f"Meta-model input shape: {X_meta.shape} (1 base pred + {X_meta_direct.shape[1]} direct features)")
        else:
            X_meta = base_predictions
            log.info(f"Meta-model input shape: {X_meta.shape} (base predictions only)")
        
        # Step 4: Train XGBoost meta-model
        log.info("Training XGBoost meta-model")
        self.meta_model.fit(X_meta, y)
        
        log.info("HybridRegressor training complete")
        return self
    
    def predict(self, X):
        """
        Generate predictions using the stacking ensemble.
        
        Steps:
            1. Get base model prediction from context features
            2. Combine with meta features (ATR, time)
            3. Pass to meta model for final prediction
        """
        X_base, X_meta_direct = self._extract_features(X)
        
        # Get base predictions
        if X_base.shape[1] > 0:
            base_predictions = self.base_model.predict(X_base).reshape(-1, 1)
        else:
            base_predictions = np.zeros((len(X), 1))
        
        # Combine with meta features
        if X_meta_direct.shape[1] > 0:
            X_meta = np.hstack([base_predictions, X_meta_direct])
        else:
            X_meta = base_predictions
        
        # Final prediction from meta model
        return self.meta_model.predict(X_meta)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'base_features': self.base_features,
            'meta_features': self.meta_features,
            'xgb_params': self.xgb_params,
            'cv_folds': self.cv_folds
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
