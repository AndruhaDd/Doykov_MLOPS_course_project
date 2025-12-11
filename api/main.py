from fastapi import FastAPI
from api.schemas import PredictRequest
from api.inference_service import run_prediction

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Air Quality Prediction API"}

@app.post("/predict")
def predict(request: PredictRequest):
    prediction = run_prediction(request.dict())
    return {"prediction": prediction}
