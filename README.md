Banking Transaction Fraud Detection & Risk Analytics Dashboard

A machine learning-powered fintech analytics system that detects fraudulent financial transactions using the PaySim synthetic banking dataset and visualizes fraud analytics through an interactive Streamlit dashboard.

  Features
Real-time fraud prediction
FinTech-style analytics dashboard
Random Forest fraud detection
Fraud probability scoring
Confusion matrix & model performance metrics
Fraud distribution visualizations
Streamlit-based UI
  Technologies Used
Python
Streamlit
Pandas
Scikit-learn
Plotly
Matplotlib
Seaborn
Joblib
  Project Structure
frauddetection/
│
├── data/
├── models/
├── app.py
├── paysim_model.py
├── requirements.txt
└── README.md
  Steps to Replicate the Project
Step 1 — Clone or Download Repository

Download the repository ZIP file or clone using Git:

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

OR manually download the ZIP.

Step 2 — Install Python

Install:

Python

Recommended Version:

Python 3.10+

While installing:
  Enable:

Add Python to PATH
Step 3 — Install Required Libraries

Open Command Prompt inside project folder.

Run:

pip install -r requirements.txt

OR manually install:

pip install streamlit pandas scikit-learn plotly matplotlib seaborn joblib
Step 4 — Download Dataset

Download the PaySim dataset from Kaggle:

PaySim Dataset (Kaggle)

Download file:

PS_20174392719_1491204439457_log.csv
Step 5 — Create Project Folders

Create the following folders inside project directory:

data/
models/
Step 6 — Place Dataset

Move the downloaded dataset into:

data/

Final path:

data/PS_20174392719_1491204439457_log.csv
Step 7 — Train the Model

Run:

python paysim_model.py

This will:

preprocess dataset
encode transaction types
train Random Forest classifier
generate fraud detection model
save trained files

Generated files:

models/paysim_model.pkl
models/type_encoder.pkl
Step 8 — Run Streamlit Dashboard

Run:

streamlit run app.py
Step 9 — Open Dashboard

Browser automatically opens:

http://localhost:8501
  Dashboard Functionalities

The dashboard includes:

Fraud analytics visualization
Transaction amount analysis
Fraud distribution charts
Real-time fraud prediction
Risk probability scoring
Model performance evaluation
Confusion matrix visualization
  Machine Learning Model

Model Used:

Random Forest Classifier

Input Features:

Transaction Step (Time)
Transaction Type
Transaction Amount
Fraud Flag Status

Target Variable:

isFraud
  Performance Metrics

The dashboard displays:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
🏦 Dataset Information

Dataset:

PaySim Synthetic Financial Dataset

The dataset simulates real-world mobile banking transactions and fraudulent financial activities.

Fraudulent behavior mainly occurs in:

TRANSFER
CASH_OUT

Transaction simulation period:

744 hours (31 days)
⚠ Notes
Raw dataset and trained model files are not uploaded due to GitHub file size limitations.
Users can regenerate the model locally using the provided training script.
