import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import dump
from yaml import safe_load
from sklearn.metrics import mean_squared_error

from preprocess import load_dataset, preprocess, split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

CONFIG_PATH = "configs/config.yaml"

# Загружаем конфиг
with open(CONFIG_PATH, "r") as f:
    config = safe_load(f)

# 1. Загружаем processed датасет
df = load_dataset(config["train"]["dataset_path"])

# 2. Предобработка (удаление пропусков и числовые признаки)
df = preprocess(df)

X_train, X_test, y_train, y_test = split(
    df,
    target_column=config["train"]["target_column"],
    test_size=config["train"]["test_size"],
    random_state=config["train"]["random_state"]
)
# 4. Выбор модели
model_type = config["train"]["model_type"]

models = {
    "linear": LinearRegression(**config["train"]["linear"]),
    "random_forest": RandomForestRegressor(**config["train"]["random_forest"]),
    "lightgbm": LGBMRegressor(**config["train"]["lightgbm"])
}

model = models[model_type]

# 5. Логирование экспериментов
mlflow.set_experiment("air_quality_experiments")

with mlflow.start_run():

    # Обучение
    model.fit(X_train, y_train)

    # Предсказания
    preds = model.predict(X_test)

    # Метрики
    mse = mean_squared_error(y_test, preds)

    # Логи
    mlflow.log_param("model_type", model_type)
    mlflow.log_metric("mse", mse)

    # Сохраняем модель
    dump(model, config["inference"]["model_path"])
    mlflow.log_artifact(config["inference"]["model_path"])

print("Модель успешно обучена и сохранена!")
