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
    model_report = dict()
    for name, model in models.items():
        param = params[name]

        gs = GridSearchCV(model, param, cv=3)
        gs.fit(X_train, y_train)

        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)

        score = r2_score(y_test, y_pred)

        model_report[name] = score
    
    return model_report