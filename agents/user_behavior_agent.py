import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class UserBehaviorAgent:

    def __init__(self):
        self.scaler = MinMaxScaler()


    def create_behavior_score(self, df):

        columns = [
            "previous_clicks",
            "page_views",
            "session_time"
        ]

        # Step 1 normalize
        normalized = self.scaler.fit_transform(df[columns])

        Click_wt = 0.4
        Page_wt = 0.3
        Session_wt = 0.3
        # Step 2 calculate score
        df["behavior_score"] = (
            (Click_wt * normalized[:,0])
            +
            (Page_wt * normalized[:,1])
            +
            (Session_wt * normalized[:,2])
        ) * 100

        print("behavior score created successfully")
        return df


if __name__ == "__main__":

    df = pd.read_csv(
        "data/raw/ctr_dataset.csv"
    )

    agent = UserBehaviorAgent()

    df = agent.create_behavior_score(df)

    print(df["previous_clicks", "page_views", "session_time","behavior_score"].head())