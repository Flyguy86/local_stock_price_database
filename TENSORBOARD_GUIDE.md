# Training Performance Visualization Guide

This guide shows you how to monitor and visualize your training runs using TensorBoard and other tools.

## Quick Start

### 1. TensorBoard (Recommended for Training Metrics) ✅

**Start TensorBoard:**
```bash
# Make the script executable (first time only)
chmod +x start_tensorboard.sh

# Start TensorBoard
./start_tensorboard.sh

# Or specify a custom port
./start_tensorboard.sh 6007
```

**Access in Browser:**
```
http://localhost:6006
```

**What You'll See:**
- **SCALARS Tab**: 
  - Grid Search CV scores for each hyperparameter combination
  - Final test metrics (R², MSE, RMSE, MAE, Accuracy, etc.)
  - Model performance comparisons across runs
  
- **TEXT Tab**:
  - Model metadata (algorithm, symbol, target)
  
- **GRAPHS Tab** (if available):
  - Model architecture visualization

**Tips:**
- Use the left sidebar to filter by experiment/run
- Compare multiple training runs side-by-side
- Smooth curves using the smoothing slider
- Download data as CSV for external analysis

### 2. Ray Dashboard (Distributed Training Monitoring) ✅

**Access:**
```
http://localhost:8265
```

**Features:**
- **Jobs**: View all training tasks and their status
- **Actors**: Monitor training workers and resource allocation
- **Metrics**: CPU, memory, object store usage in real-time
- **Logs**: Stream logs from Ray tasks
- **Timeline**: Visualize task execution and dependencies

**Use Cases:**
- Check if training jobs are stuck
- Monitor resource utilization (CPU/memory bottlenecks)
- Debug distributed training issues
- View parallel training execution

### 3. Training Service Web UI

**Access:**
```
http://localhost:8001
```

**Features:**
- Model metadata and hyperparameters
- Feature importance (SHAP, Permutation, Coefficient)
- Training job status and history
- Simulation backtest results
- Feature selection interface

## TensorBoard Deep Dive

### Logged Metrics

#### For All Models:
- **Model/Algorithm**: Model type (ElasticNet, XGBoost, etc.)
- **Model/Symbol**: Ticker symbol
- **Model/Target**: Target column (close, open, etc.)

#### For Regression Models:
- **Metrics/R2_Score**: Coefficient of determination
- **Metrics/MSE**: Mean Squared Error
- **Metrics/RMSE**: Root Mean Squared Error
- **Metrics/MAE**: Mean Absolute Error

#### For Classification Models:
- **Metrics/Accuracy**: Overall accuracy
- **Metrics/Precision**: Precision score
- **Metrics/Recall**: Recall score
- **Metrics/F1_Score**: F1 score

#### For Grid Search:
- **GridSearch/CV_Score**: Cross-validation score for each combination
- **GridSearch/ElasticNet**: Detailed scores by alpha and l1_ratio
- Similar metrics for XGBoost, LightGBM, RandomForest grids

### TensorBoard Command Line Options

```bash
# Basic usage
tensorboard --logdir=/app/data/tensorboard_logs

# Custom host and port (for remote access)
tensorboard --logdir=/app/data/tensorboard_logs --host=0.0.0.0 --port=6006

# Reload data more frequently (default: 5 seconds)
tensorboard --logdir=/app/data/tensorboard_logs --reload_interval=2

# Compare multiple runs from different directories
tensorboard --logdir_spec=run1:/path/to/run1,run2:/path/to/run2

# Bind to specific host
tensorboard --logdir=/app/data/tensorboard_logs --bind_all
```

### Log Directory Structure

```
/app/data/tensorboard_logs/
├── AAPL_elasticnet_regression_abc12345/
│   └── events.out.tfevents...
├── TSLA_xgboost_regressor_def67890/
│   └── events.out.tfevents...
└── SPY_random_forest_classifier_ghi24680/
    └── events.out.tfevents...
```

Each training run creates a subdirectory: `{symbol}_{algorithm}_{training_id[:8]}/`

## Workflow Examples

### Monitor a Single Training Job

1. Start training via API or UI
2. Open TensorBoard: `http://localhost:6006`
3. Watch metrics update in real-time
4. Check Ray Dashboard for resource usage
5. Review final results in Training Service UI

### Compare Multiple Models

1. Train multiple models with different hyperparameters
2. Open TensorBoard
3. Select multiple runs in the left sidebar
4. Compare metrics side-by-side in the SCALARS tab
5. Identify best performing configuration

### Debug Grid Search

1. Start grid search with many hyperparameter combinations
2. Open TensorBoard
3. Go to GridSearch section
4. See CV scores for each alpha/l1_ratio combination
5. Identify which parameter ranges work best

### Monitor Long-Running Jobs

1. Start background training job
2. Open Ray Dashboard: `http://localhost:8265`
3. Check CPU/memory utilization
4. View task timeline to ensure progress
5. Check TensorBoard for intermediate results

## Troubleshooting

### TensorBoard Not Starting

**Error:** `tensorboard: command not found`
```bash
# Install tensorboard
pip install tensorboard>=2.15.0

# Or in container
docker exec -it <container> pip install tensorboard
```

**Error:** `Permission denied`
```bash
# Create logs directory
mkdir -p /app/data/tensorboard_logs

# Check permissions
ls -la /app/data/
```

### No Data in TensorBoard

**Issue:** TensorBoard starts but shows no data

**Solutions:**
1. Check if training has completed:
   ```bash
   ls -la /app/data/tensorboard_logs/
   ```

2. Verify SummaryWriter is initialized (check logs):
   ```
   TensorBoard logging enabled: /app/data/tensorboard_logs/...
   ```

3. Wait for training to log metrics (metrics logged at end of training)

4. Refresh TensorBoard in browser (F5)

### Ray Dashboard Not Accessible

**Issue:** Cannot access http://localhost:8265

**Solutions:**
1. Check if Ray is running:
   ```bash
   ps aux | grep ray
   ```

2. Verify Ray dashboard port:
   ```bash
   ray status
   ```

3. Restart Ray with explicit dashboard settings:
   ```bash
   ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265
   ```

## Advanced Usage

### Export TensorBoard Data

```bash
# Export scalars to CSV
tensorboard --logdir=/app/data/tensorboard_logs --export_to_csv=output.csv

# Use Python API
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator('path/to/logs')
ea.Reload()
print(ea.Scalars('Metrics/R2_Score'))
```

### Filter by Time Range

In TensorBoard UI:
1. Click the settings gear icon
2. Adjust "Time" slider to focus on specific period
3. Use "Relative" or "Wall" time modes

### Custom Metrics

To add custom metrics to your training code:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='path/to/logs')

# Log scalar
writer.add_scalar('Custom/MyMetric', value, step)

# Log multiple scalars
writer.add_scalars('Custom/Multi', {
    'metric1': val1,
    'metric2': val2
}, step)

# Log histogram
writer.add_histogram('Weights/layer1', weight_tensor, step)

writer.close()
```

## References

- **TensorBoard Documentation**: https://www.tensorflow.org/tensorboard
- **Ray Dashboard Guide**: https://docs.ray.io/en/latest/ray-observability/ray-dashboard.html
- **PyTorch TensorBoard Tutorial**: https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
