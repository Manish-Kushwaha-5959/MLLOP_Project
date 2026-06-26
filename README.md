# MLOPS Project — Student Performance Prediction

An end-to-end **Machine Learning Operations (MLOps)** project that predicts a student's **math score** based on demographic and academic features. The project covers the full ML lifecycle — from data ingestion and transformation to model training, hyperparameter tuning, and deployment via a Flask web application.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Models Evaluated](#models-evaluated)
- [Getting Started](#getting-started)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Author](#author)
- [License](#license)

---

## Overview

This project demonstrates a production-style MLOps workflow for a regression use case. A student performance dataset is ingested, preprocessed, and used to train multiple regression models. The best-performing model is selected, serialized, and served through a Flask web interface that accepts user input and returns a predicted math score in real time.

---

## Problem Statement

Given a student's demographic and academic attributes (gender, race/ethnicity, parental education, lunch type, test preparation course, reading and writing scores), predict their **math score**.

This is a supervised **regression** problem.

---

## Dataset

- **Source:** `notebook/data/stud.csv`
- **Target variable:** `math_score`
- **Features:**

| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Student's gender |
| `race_ethnicity` | Categorical | Race/Ethnicity group |
| `parental_level_of_education` | Categorical | Parent's highest education |
| `lunch` | Categorical | Lunch type (standard / free-reduced) |
| `test_preparation_course` | Categorical | Whether the course was completed |
| `reading_score` | Numerical | Reading score (0–100) |
| `writing_score` | Numerical | Writing score (0–100) |

---

## Tech Stack

- **Language:** Python 3.8+
- **ML Libraries:** scikit-learn, XGBoost, CatBoost
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Web Framework:** Flask
- **Serialization:** pickle / dill
- **Logging:** Custom logger with timestamped log files
- **Packaging:** setuptools

---

## Project Structure

```
MMLOP_Project/
├── application.py                # Flask web application entry point
├── requirements.txt              # Python dependencies
├── setup.py                      # Package configuration
├── README.md
├── .gitignore
│
├── notebook/
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb   # Exploratory Data Analysis
│   ├── 2. MODEL TRAINING.ipynb            # Model experimentation
│   └── data/
│       └── stud.csv                       # Raw dataset
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py              # Reads & splits raw data
│   │   ├── data_transformation.py         # Preprocessing pipeline
│   │   └── model_trainer.py               # Trains & evaluates models
│   ├── pipeline/
│   │   ├── train_pipeline.py              # Training orchestration
│   │   └── predict_pipeline.py            # Inference pipeline
│   ├── utils.py                           # save/load/evaluate helpers
│   ├── logger.py                          # Timestamped logging
│   └── exception.py                       # Custom exception class
│
├── templates/
│   ├── index.html                         # Landing page
│   └── home.html                          # Prediction form
│
├── artifacts/
│   ├── raw.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl                   # Fitted preprocessing pipeline
│   └── model.pkl                          # Best trained model
│
├── logs/                                  # Timestamped runtime logs
└── catboost_info/                         # CatBoost training metadata
```

---

## ML Pipeline

The training pipeline runs in three sequential stages:

### 1. Data Ingestion (`src/components/data_ingestion.py`)
- Loads `notebook/data/stud.csv`
- Saves the raw data to `artifacts/raw.csv`
- Splits into train/test (80/20, `random_state=42`)
- Persists splits to `artifacts/train.csv` and `artifacts/test.csv`
- Returns the train/test paths as a tuple — orchestration is delegated to `src/pipeline/train_pipeline.py`

### 2. Data Transformation (`src/components/data_transformation.py`)
Builds a scikit-learn `ColumnTransformer` with two pipelines:

- **Numerical pipeline** (`reading_score`, `writing_score`)
  - Median imputation → Standard scaling
- **Categorical pipeline** (`gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`)
  - Most-frequent imputation → One-hot encoding (unknown handling) → Standard scaling (without centering)

The fitted preprocessor is saved to `artifacts/preprocessor.pkl`.

### 3. Model Training (`src/components/model_trainer.py`)
- Trains 8 regression models
- Performs hyperparameter tuning via `GridSearchCV` (cv=3, scoring=r²)
- Selects the best model based on test R² score (threshold ≥ 0.6)
- Saves the winning model to `artifacts/model.pkl`

### 4. Training Pipeline (`src/pipeline/train_pipeline.py`)
- The `TrainPipeline.start_training()` class method wires the three components together in sequence
- Run via `python src/pipeline/train_pipeline.py`, or import and call it from another script

---

## Models Evaluated

| Model | Hyperparameters Tuned |
|---|---|
| Linear Regression | — |
| K-Neighbors Regressor | — |
| Decision Tree Regressor | `criterion`, `splitter`, `max_features` |
| Random Forest Regressor | `criterion`, `max_features`, `n_estimators` |
| AdaBoost Regressor | `learning_rate`, `n_estimators` |
| Gradient Boosting Regressor | `learning_rate`, `subsample`, `n_estimators` |
| XGBoost Regressor | `learning_rate`, `n_estimators` |
| CatBoost Regressor | `depth`, `learning_rate`, `iterations` |

Selection metric: **R² score** on the test set.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Manish-Kushwaha-5959/MMLOP_Project.git
   cd MMLOP_Project
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the project as a package (optional, for `src` imports)**
   ```bash
   pip install -e .
   ```

---

## Running the Application

### 1. Train the model (optional — pre-trained artifacts are included)

Run the end-to-end training pipeline:

```bash
python src/pipeline/train_pipeline.py
```

This orchestrates all three stages:
- **Data Ingestion** (`DataIngestion`) — reads `notebook/data/stud.csv`, saves `artifacts/raw.csv`, splits 80/20 (`random_state=42`), persists `artifacts/train.csv` and `artifacts/test.csv`. Returns the train/test paths as a tuple.
- **Data Transformation** (`DataTransformation`) — fits the `ColumnTransformer` (numerical + categorical pipelines), saves `artifacts/preprocessor.pkl`, and returns transformed train/test arrays.
- **Model Training** (`ModelTrainer`) — runs `GridSearchCV` over all 8 models, picks the best by test R² (threshold ≥ 0.6), and saves `artifacts/model.pkl`.

You can also call the pipeline programmatically:

```python
from src.pipeline.train_pipeline import TrainPipeline

pipeline = TrainPipeline()
print(pipeline.start_training())
```

### 2. Launch the Flask web app

```bash
python application.py
```

The app runs with debug mode enabled at:

```
http://127.0.0.1:5000/
```

---

## Usage

1. Open `http://127.0.0.1:5000/` in your browser.
2. Click **Predict** (or go to `/predict`).
3. Fill in the form fields:
   - Gender
   - Race/Ethnicity
   - Parental Level of Education
   - Lunch type
   - Test Preparation Course
   - Reading Score (0–100)
   - Writing Score (0–100)
4. Submit the form to receive the predicted **math score**.

### Programmatic prediction

```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

data = CustomData(
    gender="female",
    race_ethnicity="group B",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="none",
    reading_score=72,
    writing_score=74,
)

predict_pipeline = PredictPipeline()
print(predict_pipeline.predict(data.get_data_as_dataframe()))
```

---

## Logging

All pipeline stages emit timestamped log files into the `logs/` directory. Each run creates a new file named `MM_DD_YYYY_HH_MM_SS.log`. Exceptions are wrapped using the project's custom `CustomException` class for consistent error reporting.

---

## Author

**Manish Kushwaha**
- Email: manishkushwaha5959@gmail.com
- GitHub: [@Manish-Kushwaha-5959](https://github.com/Manish-Kushwaha-5959)

---

## License

This project is released under the MIT License. See `LICENSE` for details (or add one if you plan to open-source it formally).
