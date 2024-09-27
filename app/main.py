from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

# Load the saved Random Forest model
with open("/app/notebooks/Random_Forest_24-09-2024-18-48-55.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Create FastAPI instance
app = FastAPI()

# Define input schema for the prediction request
class SalesPredictionInput(BaseModel):
    Store: int
    DayOfWeek: int
    Promo: int
    StateHoliday: str
    StoreType: str
    Assortment: str
    Promo2: int
    Day: int
    Month: int
    Year: int

# API endpoint to predict sales
@app.post("/predict/")
def predict_sales(input_data: SalesPredictionInput):
    # Convert input data to DataFrame for prediction
    input_df = pd.DataFrame([input_data.dict()])
    print(input_df)
    print(transformed_input)

    # Perform preprocessing using the pipeline
    try:
        transformed_input = pipeline.transform(input_df)
        prediction = model.predict(transformed_input)
        return {"Predicted Sales": prediction[0]}
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def read_root():
    return {"message": "Predictive model API is running!"}



