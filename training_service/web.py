<div class="info-box"><!DOCTYPE html>















        </div>            </ul>                </li>                    </ul>                        <li>Best for: Capturing both linear market relationships and complex patterns</li>                        <li>Meta: XGBoost on base predictions + ATR + time features</li>                        <li>Base: Linear Regression on context symbols (QQQ, MSFT, GOOGL)</li>                    <ul style="margin-left: 20px; font-size: 11px;">                <li><strong>HybridRegressor:</strong> Two-stage stacking model:                <li><strong>ElasticNet:</strong> Linear regression with L1+L2 regularization</li>                <li><strong>LightGBM:</strong> Fast gradient boosting, handles large datasets efficiently</li>                <li><strong>XGBoost:</strong> Gradient boosting, excellent for structured data</li>                <li><strong>Random Forest:</strong> Ensemble of decision trees, good for non-linear patterns</li>            <ul>            <h3>📊 Algorithm Guide</h3><html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Training Service</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.5.2/css/bootstrap-grid.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.5.2/css/bootstrap-reboot.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/select2/4.0.13/css/select2.min.css" />
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/jqueryui/1.12.1/jquery-ui.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/c3/0.4.11/c3.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/fullcalendar/3.10.2/fullcalendar.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/sweetalert/1.1.3/sweetalert.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"/>
    <link rel="stylesheet" href="https://unpkg.com/@@/dist/tailwind.min.css">
    <style>
        /* Custom styles here */
    </style>
</head>
<body class="bg-gray-100">
    <div class="container mx-auto py-8">
        <div class="bg-white p-6 rounded-lg shadow-md">
            <h2 class="text-2xl font-semibold mb-4">Train Machine Learning Model</h2>
            <form id="trainingForm">
                <div class="mb-4">
                    <label for="modelType" class="block text-gray-700 text-sm font-bold mb-2">Model Type:</label>
                    <div class="relative">
                        <select id="modelType" name="modelType" class="form-select block w-full py-2 px-3 rounded-md border border-gray-300 focus:outline-none focus:ring focus:ring-blue-200">
                            <option value="RandomForest">Random Forest</option>
                            <option value="XGBoost">XGBoost</option>
                            <option value="LightGBM">LightGBM</option>
                            <option value="ElasticNet">ElasticNet (L1+L2 Regularization)</option>
                            <option value="HybridRegressor">HybridRegressor (Stacking: Linear → XGBoost)</option>
                        </select>
                    </div>
                </div>
                <div class="mb-4">
                    <label for="trainingData" class="block text-gray-700 text-sm font-bold mb-2">Training Data:</label>
                    <input type="file" id="trainingData" name="trainingData" accept=".csv" class="form-input block w-full py-2 px-3 rounded-md border border-gray-300 focus:outline-none focus:ring focus:ring-blue-200">
                </div>
                <div class="mb-4">
                    <label for="targetColumn" class="block text-gray-700 text-sm font-bold mb-2">Target Column:</label>
                    <input type="text" id="targetColumn" name="targetColumn" class="form-input block w-full py-2 px-3 rounded-md border border-gray-300 focus:outline-none focus:ring focus:ring-blue-200">
                </div>
                <div class="flex items-center justify-between">
                    <button type="submit" class="btn btn-blue">
                        <i class="fas fa-rocket mr-2"></i> Train Model
                    </button>
                </div>
            </form>
        </div>
        <div class="mt-8">
            <div id="modelPerformance" class="bg-white p-6 rounded-lg shadow-md hidden">
                <h3 class="text-xl font-semibold mb-4">Model Performance</h3>
                <div id="performanceMetrics" class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <!-- Performance metrics will be populated here -->
                </div>
                <div class="mt-4">
                    <button id="viewResults" class="btn btn-green">
                        <i class="fas fa-chart-line mr-2"></i> View Results
                    </button>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.5.2/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/select2/4.0.13/js/select2.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jqueryui/1.12.1/jquery-ui.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/5.16.0/d3.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/c3/0.4.11/c3.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/fullcalendar/3.10.2/fullcalendar.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/sweetalert/1.1.3/sweetalert.min.js"></script>
    <script>
        $(document).ready(function() {
            // Initialize select2 for model type
            $('#modelType').select2({
                placeholder: "Select a model type",
                allowClear: true
            });

            // Handle form submission
            $('#trainingForm').on('submit', function(e) {
                e.preventDefault();
                // TODO: Implement model training logic
                const formData = new FormData(this);
                const modelType = formData.get('modelType');
                const trainingData = formData.get('trainingData');
                const targetColumn = formData.get('targetColumn');

                // Simulate model training and show performance metrics
                setTimeout(() => {
                    $('#modelPerformance').removeClass('hidden');
                    $('#performanceMetrics').html(`
                        <div class="bg-blue-50 p-4 rounded-md">
                            <p class="text-sm text-blue-700"><strong>Model Type:</strong> ${modelType}</p>
                            <p class="text-sm text-blue-700"><strong>Training Data:</strong> ${trainingData.name}</p>
                            <p class="text-sm text-blue-700"><strong>Target Column:</strong> ${targetColumn}</p>
                        </div>
                        <div class="bg-white p-4 rounded-md shadow-sm">
                            <h4 class="text-md font-semibold mb-2">Performance Metrics</h4>
                            <p class="text-sm text-gray-700"><strong>Accuracy:</strong> 0.95</p>
                            <p class="text-sm text-gray-700"><strong>Precision:</strong> 0.93</p>
                            <p class="text-sm text-gray-700"><strong>Recall:</strong> 0.92</p>
                            <p class="text-sm text-gray-700"><strong>F1 Score:</strong> 0.925</p>
                        </div>
                    `);
                }, 2000);
            });

            // Handle view results button click
            $('#viewResults').on('click', function() {
                // TODO: Implement view results logic
                swal("Results", "Here you can display the results of the model training and evaluation.", "info");
            });
        });
    </script>
</body>
</html>