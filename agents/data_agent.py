# data_agent.py

import pandas as pd


class DataAgent:

    def __init__(self, file_path):
        self.file_path = file_path


    def load_data(self):
        """
        Load CSV file
        """

        df = pd.read_csv(self.file_path)

        print("Data loaded successfully")

        return df


    def check_missing_values(self, df):
        """
        Check null values
        """

        print("\nMissing values:\n")

        print(df.isnull().sum())


    def basic_info(self, df):
        """
        Show dataset information
        """

        print("\nDataset Info:\n")

        print(df.info())


# Testing code

if __name__ == "__main__":

    agent = DataAgent(
        "data/raw/ctr_dataset.csv"
    )

    df = agent.load_data()

    agent.check_missing_values(df)

    agent.basic_info(df)