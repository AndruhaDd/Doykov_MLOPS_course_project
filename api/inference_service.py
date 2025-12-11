from src.inference import load_model, predict

# Загружаем модель один раз при старте
model = load_model("models/trained_model.pkl")

def run_prediction(data: dict):
    return predict(model, data)
