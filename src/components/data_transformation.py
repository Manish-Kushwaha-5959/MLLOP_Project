from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("creating preprocessing object")
            # define the numerical columns and categorical columns names
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # numerical column pipeline
            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # categorical column pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # merge the pipline
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Preprocessor object created successfully")

            # return the preprocessor object
            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("starting data transformation")

            # load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("train and test data loaded")

            # get preprocessing object
            preprocessor = self.get_data_transformer_object()

            target_column = "math_score"

            # split features and target
            input_features_train = train_df.drop(columns=[target_column])
            target_feature_train = train_df[target_column]

            input_features_test = test_df.drop(columns=[target_column])
            target_feature_test = test_df[target_column]

            # apply transformation
            train_arr = preprocessor.fit_transform(input_features_train)
            test_arr = preprocessor.transform(input_features_test)

            logging.info("data transformation completed")

            # save preprocessor
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            
            logging.info("Preprocessor saved successfully")

            # combine features + target
            train_final = np.c_[train_arr, target_feature_train]
            test_final = np.c_[test_arr, target_feature_test]

            return (
                train_final,
                test_final,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

