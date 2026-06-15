# feature_agent.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class FeatureAgent:

    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


    def encode_categorical_columns(self, df):

        categorical_columns = [
            "gender",
            "device_type",
            "ad_category",
            "time_of_day"
        ]

        encoded = self.encoder.fit_transform(df[categorical_columns])
        encoded_df = pd.DataFrame(encoded, columns = self.encoder.get_feature_names_out(categorical_columns))
        
        df = df.drop(columns=categorical_columns)
        df = pd.concat([df,encoded_df], axis=1)
        
        print("encoding Completed")

        return df
        
        


    def show_transformed_data(self, df):

        print("\nEncoded Dataset:\n")

        print(df.head())


# Testing

if __name__ == "__main__":

    df = pd.read_csv(
        "data/raw/ctr_dataset.csv"
    )

    agent = FeatureAgent()

    df = agent.encode_categorical_columns(df)

    agent.show_transformed_data(df)