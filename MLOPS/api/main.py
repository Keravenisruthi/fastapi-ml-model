from fastapi import FastAPI
#pydantic is used to import the model
from pydantic import BaseModel
import joblib
import os
app = FastAPI()
model_path = os.path.join(os.path.dirname(__file__),"model.pkl")
model = joblib.load(model_path)
class InputData(BaseModel):
    marks:list[float]
@app.post("/predict")
def predict(data:InputData):
    prediction=model.predict([data.marks])
    return {"prediction":int(prediction[0])}
   # return "testing"
