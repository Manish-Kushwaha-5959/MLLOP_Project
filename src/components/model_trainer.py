import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Starting model training")

            # splitting dataset
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # model list
            models = {
                "Linear Regression" : LinearRegression(),
                "KNeighbors Regression" : KNeighborsRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "AdaBoost Regressor" : AdaBoostRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "CatBoosting Regressor" : CatBoostRegressor(verbose=False),
                "XGBRegressor" : XGBRegressor()
            }

            # Parameters for hyperparameter tuning 
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['squared_error', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "KNeighbors Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            # training and getting the model report
            logging.info("Model evaluation started")
            model_report, trained_model = evaluate_model(X_train, y_train, X_test, y_test, models, params)

            # getting the best model from the model report
            best_model = max(model_report, key= model_report.get)
            best_model_score = max(model_report.values())
            logging.info(f"Best model found: {best_model} : {best_model_score}")

            if best_model_score < 0.6:
                raise Exception("no best model found")
            logging.info("Best model selected based on test R2 score")
            
            # training the best model
            model = trained_model[best_model]

            # saving the model
            logging.info("Saving trained model")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            return best_model_score
            
        except Exception as e:
            raise CustomException(e, sys)