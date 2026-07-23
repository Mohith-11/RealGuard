# Step-by-Step running guide: RealGuard Platform

This guide is designed for team members to set up and run the **RealGuard** platform on their local system from scratch.

---

## 🛠️ Step 1: Clone & Configure Workspace
First, clone the repository and navigate into the project root:
```bash
git clone https://github.com/Mohith-11/RealGuard.git
cd RealGuard
```

---

## 💾 Step 2: Download the Dataset
Because the credit card dataset is too large (>140MB) for GitHub's file limits, it is excluded from Git. You must fetch it manually:
1. Download the **Credit Card Fraud Detection** dataset from Kaggle: [Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Create the raw data directory structure in your project root:
   ```bash
   mkdir -p data/raw
   ```
3. Extract the downloaded archive and place the `creditcard.csv` file exactly at:
   ```text
   data/raw/creditcard.csv
   ```

---

## 🐍 Step 3: Setup Virtual Environment
Initialize a local Python environment and install the required dependencies:
```powershell
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install packages
pip install -r requirements.txt
```

---

## 🧠 Step 4: Run the Machine Learning Pipeline
Before launching the serving layer, you must train and register a model in your local MLflow store. 

1. Change directory to the `ml/` folder:
   ```bash
   cd ml
   ```
2. Execute the training script:
   ```bash
   python train.py
   ```
   *What this script does:*
   * Preprocesses `creditcard.csv`.
   * Trains both XGBoost and Random Forest models.
   * Logs hyperparameter grids and metrics to a local SQLite database (`mlflow.db`).
   * Evaluates performance and registers the best model under the name `RealGuard-FraudDetector`.
   * Dynamically tags the registered model version with the `"Production"` alias.
3. Return to the project root:
   ```bash
   cd ..
   ```

---

## 🐳 Step 5: Start Docker Services
Start the entire containerized orchestration layer in the background:
```bash
docker compose build
docker compose up -d
```

### What is running:
* **`realguard-api`** (`http://localhost:8000`): FastAPI server loading the `"Production"` model alias from your local SQLite database using internal symlinks. Check out Swagger docs at [http://localhost:8000/docs](http://localhost:8000/docs).
* **`mlflow-ui`** (`http://localhost:5000`): The MLflow dashboard tracking experiments, metrics, and logged models.
* **`kafka`** (`localhost:9092` / container `kafka:29092`): Kafka KRaft broker serving the messaging topics.
* **`kafka-ui`** (`http://localhost:8080`): The dashboard to inspect Kafka topics and streams.

---

## 📡 Step 6: Launch the Real-Time Streaming Pipeline
To run the decoupled messaging pipeline, open two separate terminal tabs with your virtual environment active in both:

### Terminal A: Start the Fraud Consumer
This script consumes features from the `transactions` topic, calls the containerized FastAPI prediction endpoint on the fly, and publishes results back to the `predictions` topic:
```bash
python kafka/consumer.py
```

### Terminal B: Start the Transaction Simulator
This script reads rows from your raw dataset, strips the target labels, and streams them to the `transactions` Kafka topic once every second:
```bash
python kafka/simulator.py
```

---

## 📊 Step 7: Monitor Streams
1. Open your browser and navigate to **Kafka UI** at [http://localhost:8080](http://localhost:8080).
2. Go to **Topics** -> **transactions** or **predictions**.
3. Select the **Messages** tab and click **Seek to End** or **Live Details** to watch the transactions and prediction labels stream in real-time!
