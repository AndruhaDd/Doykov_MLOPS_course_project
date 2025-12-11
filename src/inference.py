import pandas as pd
import joblib


def load_model(model_path: str):
    """
    Загружаем модель при старте API.
    """
    return joblib.load(model_path)


def predict(model, data: dict):
    """
    Превращаем входные данные в DataFrame и вызываем предсказание модели.
    """

    df = pd.DataFrame([data])

    # Переименовываем PM2_5 → PM2.5
    df = df.rename(columns={
        "PM2_5": "PM2.5"
    })

    # Заполняем отсутствующие столбцы нулями
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    # Оставляем только те фичи, которые были на обучении
    df = df[model.feature_names_in_]

    return float(model.predict(df)[0])
