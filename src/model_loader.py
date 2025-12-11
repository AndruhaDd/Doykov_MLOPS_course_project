import pandas as pd
import joblib
from preprocess import preprocess
from utils import enforce_column_order


class ModelLoader:
    """
    Класс, который загружает модель и делает предобработку данных
    так же, как это было на этапе обучения.
    """

    def __init__(self, model_path: str, feature_list_path: str):
        # Загружаем модель
        self.model = joblib.load(model_path)

        # Загружаем список фичей (важно для правильного порядка)
        with open(feature_list_path, "r") as f:
            self.feature_list = [line.strip() for line in f.readlines()]

    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Полная предобработка данных для инференса:
        1) те же шаги, что в preprocess()
        2) гарантируем порядок колонок как при обучении
        """
        df = preprocess(df)
        df = enforce_column_order(df, self.feature_list)
        return df

    def predict(self, df: pd.DataFrame):
        """
        Возвращает предсказания модели
        """
        processed = self.preprocess_input(df)
        predictions = self.model.predict(processed)
        return predictions
