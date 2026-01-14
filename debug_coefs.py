#!/usr/bin/env python3
"""
Debug script to understand feature importance extraction issue.
Simulates the training pipeline and checks what happens during coefficient extraction.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

# Create sample data similar to what we have
np.random.seed(42)
n_samples = 1000

# Create sample features similar to the real ones
data = {
    'regime_vix': np.random.randint(0, 3, n_samples),  # Regime feature (will be OneHot encoded)
    'regime_gmm': np.random.randint(0, 2, n_samples),  # Another regime
    'volume': np.random.rand(n_samples) * 1000000,  # Volume (will be RobustScaled)
    'trade_count': np.random.rand(n_samples) * 5000,  # Count (will be RobustScaled)
    'rsi': np.random.rand(n_samples) * 100,  # Bounded oscillator (passthrough)
    'ibs': np.random.rand(n_samples),  # Another bounded
    'log_return': np.random.randn(n_samples) * 0.01,  # Standard scaled
    'volatility': np.random.rand(n_samples) * 0.05,  # Standard scaled
    'macd': np.random.randn(n_samples) * 0.001,  # Standard scaled
}

df = pd.DataFrame(data)
X = df.copy()
y = np.random.randn(n_samples) * 0.01  # Target

print("=" * 80)
print("Original Features:")
print(f"Columns: {list(X.columns)}")
print(f"Shape: {X.shape}")
print()

# Replicate the preprocessing logic from trainer.py
feature_cols_used = list(X.columns)

# Categorize features (same logic as trainer.py lines 410-440)
is_linear = True  # ElasticNet is linear
cols_regime = [c for c in X.columns if c in ["regime_vix", "regime_gmm"]]
cols_robust = []
cols_passthrough = []
cols_standard = []

remaining_cols = [c for c in X.columns if c not in cols_regime]

for c in remaining_cols:
    cl = c.lower()
    if "volume" in cl or "count" in cl or "pro_vol" in cl:
        cols_robust.append(c)
    elif any(x in cl for x in ["rsi", "ibs", "aroon", "stoch", "bop", "mfi", "willr", "ultosc"]):
        cols_passthrough.append(c)
    else:
        cols_standard.append(c)

print("Scaling Groups:")
print(f"  Regimes ({len(cols_regime)}): {cols_regime}")
print(f"  Robust ({len(cols_robust)}): {cols_robust}")
print(f"  Passthrough ({len(cols_passthrough)}): {cols_passthrough}")
print(f"  Standard ({len(cols_standard)}): {cols_standard}")
print()

# Build transformers (same as trainer.py lines 458-478)
transformers = []

if cols_regime:
    if is_linear:
        transformers.append(('regime_ohe', OneHotEncoder(handle_unknown='ignore'), cols_regime))
    else:
        transformers.append(('regime_pass', 'passthrough', cols_regime))

if cols_robust:
    transformers.append(('robust', RobustScaler(), cols_robust))

if cols_standard:
    transformers.append(('standard', StandardScaler(), cols_standard))

if cols_passthrough:
    transformers.append(('bounded', 'passthrough', cols_passthrough))

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder='passthrough'
)

# Create pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', ElasticNet(max_iter=2000))
])

# Grid search (same as trainer.py)
grid_params = {
    'model__alpha': [0.0001, 0.001, 0.01],
    'model__l1_ratio': [0.1, 0.5, 0.9]
}

print("Running GridSearchCV...")
grid_search = GridSearchCV(
    model,
    grid_params,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X, y)
best_model = grid_search.best_estimator_

print(f"Best params: {grid_search.best_params_}")
print()

# NOW: Extract coefficients (replicate trainer.py lines 617-698)
print("=" * 80)
print("COEFFICIENT EXTRACTION:")
print()

estimator = best_model.named_steps['model']
preprocessor = best_model.named_steps['preprocessor']

if hasattr(estimator, 'coef_'):
    coefs = estimator.coef_
    if coefs.ndim > 1:
        coefs = coefs[0]
    
    print(f"Raw coefficient array shape: {coefs.shape}")
    print(f"Non-zero coefficients: {np.count_nonzero(coefs)}")
    print(f"Max abs coefficient: {np.abs(coefs).max():.6f}")
    print(f"Coefficients: {coefs}")
    print()
    
    # Check expanded names
    if hasattr(preprocessor, 'get_feature_names_out'):
        expanded_names = preprocessor.get_feature_names_out()
        print(f"Expanded feature names ({len(expanded_names)}):")
        for i, name in enumerate(expanded_names):
            print(f"  [{i}] {name}: coef={coefs[i]:.6f}")
        print()
        
        # Try the mapping logic from trainer.py
        feature_details = {col: {} for col in feature_cols_used}
        
        if len(expanded_names) == len(coefs):
            print("Attempting to map expanded features back to original...")
            mapped_count = 0
            unmapped_features = []
            
            for i, expanded_name in enumerate(expanded_names):
                matched = False
                for orig_col in feature_cols_used:
                    if orig_col in expanded_name:
                        if "coefficient" not in feature_details[orig_col]:
                            feature_details[orig_col]["coefficient"] = 0.0
                        feature_details[orig_col]["coefficient"] += float(coefs[i])
                        mapped_count += 1
                        print(f"  [{i}] {expanded_name} -> {orig_col} (coef={coefs[i]:.6f})")
                        matched = True
                        break
                
                if not matched:
                    unmapped_features.append((i, expanded_name, coefs[i]))
            
            print()
            print(f"Mapped {mapped_count}/{len(expanded_names)} coefficients")
            
            if unmapped_features:
                print(f"\nUnmapped features ({len(unmapped_features)}):")
                for idx, name, coef in unmapped_features:
                    print(f"  [{idx}] {name}: coef={coef:.6f}")
            
            print()
            print("Final feature importance by original column:")
            for col in feature_cols_used:
                coef_val = feature_details[col].get("coefficient", 0.0)
                print(f"  {col}: {coef_val:.6f}")
        else:
            print(f"ERROR: Dimension mismatch! expanded_names ({len(expanded_names)}) != coefs ({len(coefs)})")
    else:
        print("ERROR: preprocessor doesn't have get_feature_names_out()")
else:
    print("ERROR: estimator doesn't have coef_ attribute")

print()
print("=" * 80)
print("INVESTIGATION COMPLETE")
