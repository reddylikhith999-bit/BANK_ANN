# 🏦 Bank Term Deposit Prediction (ANN)

An end-to-end Machine Learning web application built with **Streamlit** and an **Artificial Neural Network (ANN)** (`MLPClassifier`) that predicts whether a bank customer will subscribe to a term deposit based on direct marketing campaign data.

---

## 🚀 Features
* **Artificial Neural Network (ANN):** Utilizes a Multi-Layer Perceptron (`MLPClassifier`) with a multi-layered architecture (`64, 32` hidden layers), ReLU activation, and Adam optimization.
* **Epoch Tracking:** Includes live epoch-by-epoch loss printing (`verbose=True`) during local model training.
* **Automated Preprocessing Pipeline:** Integrates a `ColumnTransformer` for automated One-Hot Encoding of categorical variables and `StandardScaler` for numerical feature normalization.
* **Interactive Streamlit Web App:** Provides a clean user interface via sidebars and input forms for real-time predictions.
* **Cloud Model Integration:** Dynamically downloads serialized model artifacts (`model_ann.pkl`) at runtime from cloud storage to comply with repository size limits.

---

## 🛠️ Tech Stack
* **Python** (3.8+)
* **scikit-learn** (Neural network modeling & preprocessing pipeline)
* **Pandas & NumPy** (Data wrangling & cleaning)
* **Streamlit** (Interactive frontend web application)
* **Joblib** (Model persistence)
* **gdown** (Secure Google Drive asset retrieval)

---

## ⚙️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/reddylikhith999-bit/BANK_ANN.git](https://github.com/reddylikhith999-bit/BANK_ANN.git)
   cd BANK_ANN
