from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException

import sys

class TrainPipeline:
    def __init__(self):
        pass

    def start_training(self):
        try:
            logging.info("Training pipeline started")

            # step-1
            ingestion = DataIngestion()

            train_path, test_path = ingestion.initiate_data_ingestion()

            logging.info("Data ingestion completed")

            # step-2
            transformation = DataTransformation()

            train_array, test_array, _ = transformation.initiate_data_transformation(train_path=train_path, test_path=test_path)

            logging.info("Data transformation completed")

            # step-3
            trainer = ModelTrainer()

            score = trainer.initiate_model_trainer(train_array=train_array, test_array=test_array)

            logging.info("Model training completed")
            return score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()

    print(pipeline.start_training())