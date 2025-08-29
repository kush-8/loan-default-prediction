# 📊 Loan Default Risk Prediction

An **end-to-end machine learning application** to predict loan default probability, built with a production-ready **MLOps workflow** — from data ingestion and EDA to model training, testing, and containerized deployment with FastAPI and Docker.

---

## 🏆 Final Model Performance
- **Final Tuned Model ROC AUC:** `0.7748`

---

## 🚀 Tech Stack
- **Languages:** Python 3.10  
- **Libraries:** Pandas, Scikit-learn, LightGBM, Optuna, SHAP, FastAPI, Pytest  
- **Tools:** Docker, Git, GitHub Actions  

---

## 📂 Project Structure
```
├── .github/         # CI/CD workflows for GitHub Actions
├── config/          # Configuration files and model parameters
├── data/            # Raw and processed datasets
├── models/          # Trained models, pipelines, and scalers
├── notebooks/       # Jupyter notebooks for EDA and experimentation
├── reports/         # Model explainability plots (e.g., SHAP)
├── scripts/         # Automation scripts (training, prediction, deployment)
├── src/             # All source code for the application
├── tests/           # Test suite for data, model, and API
├── .dockerignore    # Specifies files to ignore in the Docker build
├── .gitignore       # Specifies files for Git to ignore
├── Dockerfile       # Instructions for building the Docker container
├── README.md        # Project documentation
├── requirements-api.txt # Lean dependencies for the prediction API
└── requirements.txt # Dependencies for the full project (including training)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/kush-8/loan-default-prediction.git
cd loan-default-prediction
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

The project includes automation scripts to streamline the workflow.

### 🔹 Run the Full Pipeline (Data Ingestion + Training)
```bash
./scripts/run_pipeline.sh
```

### 🔹 Train Model Only (if data is already downloaded)
```bash
./scripts/train.sh
```

### 🔹 Batch Prediction (generate `submission.csv`)
```bash
./scripts/predict.sh
```

### 🔹 Run API with Deployment Script
Deploy the FastAPI service inside a container using the helper script:
```bash
./scripts/deploy.sh
```
The interactive API docs will be available at:  
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🐳 Docker Usage

You can also build and run the Docker image manually.

### 🔹 Build Docker Image
```bash
docker build -t loan-default-prediction .
```

### 🔹 Run Docker Container
```bash
docker run -d -p 8000:8000 loan-default-prediction
```

### 🔹 Access API
Once the container is running, the API will be available at:  
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## ▶️ Testing

Run the test suite (data validation, model testing, and API testing):
```bash
pytest
```

---

## 📊 Model Explainability

Generate SHAP plots to explain model predictions:
```bash
python src/explain.py
```
Plots will be saved in the `reports/` directory.

---

## 🔄 CI/CD Pipeline

This project uses **GitHub Actions** for continuous integration.  
Workflow: `.github/workflows/ci.yml`

On every push or pull request to `main`, it:  
- Installs dependencies  
- Runs the test suite (`pytest`)  
- Builds the Docker image  

---

## ✨ Features
- ✅ End-to-end ML pipeline (EDA → Training → Prediction → Deployment)  
- ✅ Hyperparameter tuning with **Optuna**  
- ✅ Model explainability with **SHAP**  
- ✅ API deployment with **FastAPI + Docker**  
- ✅ CI/CD automation with **GitHub Actions**  
- ✅ Modular, production-ready **MLOps structure**  
- ✅ Comprehensive **testing framework** for data, models, and API  

---

## 🛠️ Troubleshooting
- If Docker build fails:
  ```bash
  docker system prune -af
  docker build -t loan-default-prediction .
  ```
- If container does not start, check logs:
  ```bash
  docker logs <container_id>
  ```
- If virtual environment issues occur, delete `venv/` and recreate it.  
- Ensure **Python 3.10** is installed:  
  ```bash
  python --version
  ```

---

## 👥 Contributors
- **[Kush](https://github.com/kush-8)** – Developer & Maintainer  

---

## 📜 License
This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it.

---
