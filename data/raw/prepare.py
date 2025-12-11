import pandas as pd
import os

RAW_PATH = "data/raw/city_day.csv"
PROCESSED_PATH = "data/processed/air_quality.csv"

def prepare_data():

    print("Загружаю сырые данные...")
    df = pd.read_csv(RAW_PATH)

    # Удаляем строки с пропусками
    df = df.dropna()

    # Оставляем только числовые колонки + AQI
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df = df[numeric_cols]

    # Удаляем полностью дублирующиеся строки (если есть)
    df = df.drop_duplicates()


    # Сохраняем
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"Готово! Файл сохранён: {PROCESSED_PATH}")
    print("Размер:", df.shape)

if __name__ == "__main__":
    prepare_data()
