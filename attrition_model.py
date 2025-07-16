import pandas as pd
import joblib

def predict_attrition(new_data_path, model_path):
    df = pd.read_csv(new_data_path)
    model = joblib.load(model_path)
    predictions = model.predict(df)
    df['Attrition_Prediction'] = predictions
    df.to_csv("data/attrition_predictions.csv", index=False)

if __name__ == "__main__":
    predict_attrition("data/hr_dataset.csv", "data/attrition_model.pkl")
