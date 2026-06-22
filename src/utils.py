import pickle
import os
import sys
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        model_report = dict()
        trained_model = dict()
        for name, model in models.items():
            param = params[name]

            gs = GridSearchCV(model, param, cv=3, scoring="r2")
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            score = r2_score(y_test, y_pred)

            model_report[name] = score
            trained_model[name] = model
        
        return model_report, trained_model
    
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)