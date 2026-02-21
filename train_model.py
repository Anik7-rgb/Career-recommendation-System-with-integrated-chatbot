import pandas as pd
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
import joblib

# STEP 1: Load Dataset
dataset = load_dataset("Suriyaganesh/54k-resume")
skills_df = pd.DataFrame(dataset["train"]["person_skills"])
people_df = pd.DataFrame(dataset["train"]["people"])

# STEP 2: Merge & Prepare Skill Lists per person
skills_grouped = skills_df.groupby("person_id")["skill"].apply(list).reset_index()
merged_df = people_df.merge(skills_grouped, on="person_id")
merged_df = merged_df[["person_id", "skill", "job_title"]].dropna()

# STEP 3: Preprocess labels and features
X_skills = merged_df["skill"]
y_roles = merged_df["job_title"].apply(lambda x: x.strip().lower())

# STEP 4: Convert skills to binary features
mlb = MultiLabelBinarizer()
X_encoded = mlb.fit_transform(X_skills)

# STEP 5: Train Model
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_roles, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# STEP 6: Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print("✅ Model accuracy:", accuracy)

# STEP 7: Save Model and Encoder
joblib.dump(model, "model.joblib")
joblib.dump(mlb, "encoder.joblib")
