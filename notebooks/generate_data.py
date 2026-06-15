import pandas as pd
import numpy as np
import os


# -----------------------------
# Step 1: Make results reproducible
# -----------------------------
# If we use the same seed, random values remain same every run
np.random.seed(42)


# -----------------------------
# Step 2: Create folder automatically
# -----------------------------
# If data/raw does not exist, create it
os.makedirs("data/raw", exist_ok=True)


# -----------------------------
# Step 3: Define number of rows
# -----------------------------
num_rows = 5000


# -----------------------------
# Step 4: Generate synthetic CTR dataset
# -----------------------------
data = {

    # Unique user ID
    "user_id": range(1, num_rows + 1),

    # Random age between 18 and 60
    "age": np.random.randint(
        18,
        60,
        num_rows
    ),

    # Random gender
    "gender": np.random.choice(
        ["Male", "Female"],
        num_rows
    ),

    # Device type user is browsing from
    "device_type": np.random.choice(
        ["Mobile", "Desktop", "Tablet"],
        num_rows
    ),

    # Ad category shown to user
    "ad_category": np.random.choice(
        ["Electronics", "Fashion", "Food", "Travel"],
        num_rows
    ),

    # Number of historical clicks
    "previous_clicks": np.random.randint(
        0,
        20,
        num_rows
    ),

    # Number of pages visited in session
    "page_views": np.random.randint(
        1,
        30,
        num_rows
    ),

    # Time spent on website (seconds)
    "session_time": np.random.randint(
        10,
        600,
        num_rows
    ),

    # Time when user visited
    "time_of_day": np.random.choice(
        ["Morning", "Afternoon", "Night"],
        num_rows
    ),

    # Target variable (Clicked or not)
    # 70% no click, 30% clicked
    "clicked": np.random.choice(
        [0, 1],
        num_rows,
        p=[0.7, 0.3]
    )
}


# -----------------------------
# Step 5: Convert dictionary → DataFrame
# -----------------------------
df = pd.DataFrame(data)


# -----------------------------
# Step 6: Save dataset as CSV
# -----------------------------
df.to_csv(
    "data/raw/ctr_dataset.csv",
    index=False
)


# -----------------------------
# Step 7: Print confirmation
# -----------------------------
print("Dataset created successfully!")
print("\nFirst 5 rows:\n")
print(df.head())