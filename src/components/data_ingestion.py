from src.logger import logging
from src.exception import CustomException
import os
import sys
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("starting data ingestion")

        try:
            # read data
            DATA_PATH = os.path.join("notebook", "data", "stud.csv")
            df = pd.read_csv(DATA_PATH)
            logging.info("Data loaded successfully")

            # create artifacts folder
            os.makedirs("artifacts", exist_ok=True)

            # save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            # train test split
            train, test = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("train test split completed")

            # save split data
            train.to_csv(self.ingestion_config.train_data_path, index=False)
            test.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")

            # return paths
            return {
                "train_path" : self.ingestion_config.train_data_path,
                "test_path" : self.ingestion_config.test_data_path
            }
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    print(obj.initiate_data_ingestion())    