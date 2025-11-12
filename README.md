🧠 IoT Sensor Time Series Anomaly Detection
📘 Project Overview

This project implements an AI/ML-based anomaly detection system for IoT sensor data collected from industrial machinery (using the NASA Bearing Dataset).
The objective is to identify unusual sensor readings that may indicate equipment malfunction or the need for maintenance.

The solution includes:

Data exploration and cleaning

Feature engineering for time-series signals

Two anomaly detection approaches:

Isolation Forest (Unsupervised Statistical)

Autoencoder (Deep Learning)

Visual and quantitative evaluation of detected anomalies

🗂️ Folder Structure
├── IoT_Anomaly_Detection.ipynb      # Jupyter Notebook with full implementation
├── IoT_Anomaly_Detection_Report.docx # Summary report (2-3 pages)
├── README.md                         # This file
├── dataset/                          # Folder containing sensor files
│   ├── 1st_test/
│   ├── 2nd_test/
│   ├── 3rd_test/
│   └── Readme Document for IMS Bearing Data.pdf

⚙️ Requirements

Install dependencies before running the notebook:

pip install numpy pandas matplotlib seaborn scikit-learn tensorflow kagglehub

▶️ How to Run

Download dataset automatically

import kagglehub
path = kagglehub.dataset_download("vinayak123tyagi/bearing-dataset")
print(path)


Open the Jupyter Notebook

jupyter notebook IoT_Anomaly_Detection.ipynb


Run cells sequentially:

Step 1: Data loading & exploration

Step 2: Feature engineering (rolling mean, std, FFT, etc.)

Step 3: Train Isolation Forest

Step 4: Train Autoencoder model

Step 5: Compare performance and visualize anomalies

🧩 Models Implemented
1. Isolation Forest

Detects anomalies based on how isolated a data point is.

Suitable for unsupervised time-series anomaly detection.

Key Hyperparameters:

n_estimators = 200

contamination = 0.02

max_samples = 256

2. Autoencoder Neural Network

Learns to reconstruct normal time-series behavior.

Anomalies are detected when reconstruction error exceeds a threshold.

Architecture:

Encoder → Dense(128 → 64 → 32)

Decoder → Dense(32 → 64 → 128 → Output)

Optimizer: Adam, Loss: MSE

📊 Evaluation & Results
Model	Type	Pros	Detection Insight
Isolation Forest	Statistical	Fast, unsupervised	Good baseline for unseen anomalies
Autoencoder	Deep Learning	Captures temporal dependencies	More accurate but computationally heavier

Validation Method: Visual inspection & reconstruction error analysis (due to absence of labeled anomalies).

Visualization:

Line plots showing normal vs anomalous sensor readings

Error distribution histograms

Time-series reconstruction overlay

💡 Key Findings

Both models effectively detected vibration spikes indicating bearing degradation.

Autoencoder outperformed Isolation Forest in identifying subtle early anomalies.

Rolling mean & standard deviation features improved anomaly distinction.

⚠️ Limitations

Lack of labeled anomalies prevents quantitative accuracy metrics (precision/recall).

Deep learning model requires more computational power.

Sensor drift and environmental noise can cause false positives.

🚀 Future Improvements

Integrate real-time streaming anomaly detection using Kafka + TensorFlow Serving.

Implement LSTM Autoencoder for sequence-based detection.

Build a dashboard for real-time visualization and alerts.

👤 Author
