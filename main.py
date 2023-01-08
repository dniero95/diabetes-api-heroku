

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle as pk
import json
import uvicorn

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    Pregnancies :int
    Glucose :int
    BloodPressure :int
    SkinThickness :int
    Insulin :int
    BMI :float
    DiabetesPedigreeFunction :float
    Age :int

# load the save model

diabetes_model = pk.load(open('diabetes_model.sav', 'rb'))

@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters:model_input):
    input_data=input_parameters.json()
    input_dictionary = json.loads(input_data)

    preg = input_dictionary['Pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']

    input_list = [
                    preg,
                    glu,
                    bp,
                    skin,
                    insulin,
                    bmi,
                    dpf,
                    age]

    prediction = diabetes_model.predict([input_list])

    if prediction[0] == 0:
        return "person is not diabetic"
    return "person is diabetic"




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=os.getenv("PORT", default=5000), log_level="info")













