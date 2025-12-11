

# README — Air Quality Prediction Service

## 1. Назначение проекта

Проект реализует полный цикл MLOps для задачи предсказания качества воздуха (AQI)
на основе измерений концентраций загрязняющих веществ.

Сервис включает:

- подготовку данных,
- обучение и версионирование моделей,
- деплой модели в FastAPI,
- упаковку в Docker-контейнер.

---

## 2. Основной функционал

- Обучение Regression-модели (RandomForest) для предсказания загрязнённости воздуха.
- REST API сервис для инференса (/predict, /health).
- Версионирование экспериментов с использованием MLflow.
- Контейнеризация и воспроизводимость запуска.

---

## 3. Структура репозитория

mlops-course-project
--api/                 # FastAPI сервис
------ main.py
------ schemas.py
------ inference_service.py
-- src/                  # Обучение и логика модели
------ train.py
------ preprocess.py
------ inference.py
-- data/                 # Данные
------ raw
------ processed
-- models/
------ trained_model.pkl
-- configs/
------ config.yaml
-- mlruns/                # Логи MLflow
-- Dockerfile
-- requirements.txt
-- MODEL_CARD.md
-- DATASET_CARD.md


---

## 4. Запуск сервиса

Сборка Docker-образа: docker build -t air-quality-inference .
Запуск контейнера: docker run -p 8000:8000 air-quality-inference

API доступен по адресу: http://localhost:8000

Документация Swagger: http://localhost:8000/docs

---

## 5. Эндпоинты

/health - Проверка, что сервис работает.

/predict - Принимает JSON из 12 параметров загрязняющих веществ.

Пример запроса:

{
  "PM2_5": 81.40,
  "PM10": 124.50,
  "NO": 1.44,
  "NO2": 20.50,
  "NOx": 12.08,
  "NH3": 10.72,
  "CO": 0.12,
  "SO2": 15.24,
  "O3": 127.09,
  "Benzene": 0.20,
  "Toluene": 6.50,
  "Xylene": 0.06
}


Ответ:

{
  "prediction": 186.9
}

---

## 6. Обучение модели
Запуск обучения: python src/train.py


В процессе:
- данные загружаются и очищаются
- формируется train/test
- тренируется RandomForest
- логируется MSE

сохраняется модель: models/trained_model.pkl

---

## 7. Эксперименты и версии (MLflow)

Проект использует MLflow для:
- логирования метрик
- сравнения моделей
- хранения артефактов
- хранения версий данных и препроцессинга


Запуск MLflow UI: mlflow ui

Доступно по адресу: http://127.0.0.1:5000

---

## 8. Конфигурации

Все параметры обучения и инференса вынесены в configs/config.yaml.

Пример секции:

train:
  dataset_path: data/processed/air_quality.csv
  target_column: PM2.5
  model_type: random_forest
  test_size: 0.2
  random_state: 42




## 9. CI/CD

Реализован GitHub Actions workflow (ci.yml):
- сборка Docker-образа
- тест запуска контейнера
- healthcheck

Pipeline гарантирует воспроизводимость и корректность сборки.



## 10. Пайплайн проекта

RAW data
   ↓
prepare.py
   ↓
Processed data
   ↓
train.py (MLflow)
   ↓
trained_model.pkl
   ↓
Docker build
   ↓
FastAPI inference service













