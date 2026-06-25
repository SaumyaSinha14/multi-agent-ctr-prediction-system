from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Load saved model
model = joblib.load("models/best_model.pkl")

class CTRInput(BaseModel):

    user_id: int
    age: int
    previous_clicks: int
    page_views: int
    session_time: int

    gender: str
    device_type: str
    ad_category: str
    time_of_day: str
    
@app.get("/")
def home():

    return {

        "message": "CTR Prediction API Running Successfully"

    }


@app.post("/predict")
def predict(data: CTRInput):

    # Create base dataframe

    sample_data = {

        "user_id": data.user_id,
        "age": data.age,
        "previous_clicks": data.previous_clicks,
        "page_views": data.page_views,
        "session_time": data.session_time
    }

    # Create behavior score

    behavior_score = (

        data.previous_clicks * 0.4 +

        data.page_views * 0.3 +

        data.session_time * 0.3

    )

    sample_data["behavior_score"] = behavior_score


    # Gender encoding

    sample_data["gender_Female"] = 1 if data.gender == "Female" else 0
    sample_data["gender_Male"] = 1 if data.gender == "Male" else 0


    # Device encoding

    sample_data["device_type_Desktop"] = 1 if data.device_type == "Desktop" else 0
    sample_data["device_type_Mobile"] = 1 if data.device_type == "Mobile" else 0
    sample_data["device_type_Tablet"] = 1 if data.device_type == "Tablet" else 0


    # Ad category encoding

    sample_data["ad_category_Electronics"] = 1 if data.ad_category == "Electronics" else 0
    sample_data["ad_category_Fashion"] = 1 if data.ad_category == "Fashion" else 0
    sample_data["ad_category_Food"] = 1 if data.ad_category == "Food" else 0
    sample_data["ad_category_Travel"] = 1 if data.ad_category == "Travel" else 0


    # Time encoding

    sample_data["time_of_day_Afternoon"] = 1 if data.time_of_day == "Afternoon" else 0
    sample_data["time_of_day_Morning"] = 1 if data.time_of_day == "Morning" else 0
    sample_data["time_of_day_Night"] = 1 if data.time_of_day == "Night" else 0


    # Match exact training column order

    column_order = [

        "user_id",
        "age",
        "previous_clicks",
        "page_views",
        "session_time",

        "gender_Female",
        "gender_Male",

        "device_type_Desktop",
        "device_type_Mobile",
        "device_type_Tablet",

        "ad_category_Electronics",
        "ad_category_Fashion",
        "ad_category_Food",
        "ad_category_Travel",

        "time_of_day_Afternoon",
        "time_of_day_Morning",
        "time_of_day_Night",

        "behavior_score"
    ]


    df = pd.DataFrame([sample_data])

    df = df[column_order]


    prediction = model.predict(df)


    return {

        "prediction": int(prediction[0])

    }
    