from fastapi import FastAPI
import pandas as pd

# Import your prediction function and the new schema
from .predict import make_single_prediction
from .api_schema import LoanApplication

app = FastAPI(title="Loan Default Prediction API", version="1.0")

@app.post("/predict")
def predict_default(application_data: LoanApplication):
    input_df = pd.DataFrame([application_data.dict(exclude_unset=True)])
    probability = make_single_prediction(input_df)
    return {"default_probability": f"{probability:.4f}"}

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running"}