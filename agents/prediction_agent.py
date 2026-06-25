import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.metrics import precision_score, recall_score
from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.user_behavior_agent import UserBehaviorAgent
import joblib

class PredictionAgent:

    def train_models(self, df):

        # Step 1 separate features and target

        X = df.drop("clicked", axis=1)

        y = df["clicked"]


        # Step 2 split data

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size = 0.2,
            random_state = 42,
            stratify=y
        )


        # Step 3 Logistic Regression

        lr_model = LogisticRegression(max_iter=1000)

        lr_model.fit(X_train,y_train)

        lr_predictions = lr_model.predict(X_test)


        # Step 4 XGBoost

        xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

        xgb_model.fit(X_train,y_train)

        xgb_predictions = xgb_model.predict(X_test)


        # Step 5 compare accuracy

        lr_accuracy = accuracy_score(y_test,lr_predictions)

        xgb_accuracy = accuracy_score(y_test,xgb_predictions)

        lr_precision = precision_score(y_test,lr_predictions,zero_division=0)
        xgb_precision = precision_score(y_test,xgb_predictions,zero_division=0)

        lr_recall = recall_score(y_test,lr_predictions)
        xgb_recall = recall_score(y_test,xgb_predictions)

        print("Logistic Regression Accuracy:",
              lr_accuracy)

        print("XGBoost Accuracy:",
              xgb_accuracy)

        
        cm1 = confusion_matrix(y_test, lr_predictions)
        print("Logistic Regression Confusion Matrix \n")
        print(cm1)
        cm2 = confusion_matrix(y_test,xgb_predictions)
        print("XG Boost Confusion Matrix \n")
        print(cm2)
        
        print("Logistic Regression Classification Report \n")
        print(classification_report(y_test,lr_predictions,zero_division=0))
        print("\n")
        print("XG Boost Classification Report\n")
        print(classification_report(y_test, xgb_predictions,zero_division=0))


        # Returning results to langGraph
        return {
            "metrics": 
            
            {

                 "lr_accuracy": lr_accuracy,

                 "xgb_accuracy": xgb_accuracy,

                 "lr_precision": lr_precision,

                 "xgb_precision": xgb_precision,

                 "lr_recall": lr_recall,

                 "xgb_recall": xgb_recall
    
            },

            "models": { 
                
                 "lr_model": lr_model,

                 "xgb_model": xgb_model
            
            }

        }



if __name__ == "__main__":

    data_agent = DataAgent()

    feature_agent = FeatureAgent()

    behavior_agent = UserBehaviorAgent()
    
    df = data_agent.load_data()

    df = feature_agent.encode_categorical_columns(df)

    df = behavior_agent.create_behavior_score(df)

    print(df.columns.tolist())

    agent = PredictionAgent()

    agent.train_models(df)