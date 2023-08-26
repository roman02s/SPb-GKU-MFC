from fastapi import FastAPI
from pydantic import BaseModel

from ru_rag.serve import answer, populate_db


# Определяем модель данных для входных параметров
class InputData(BaseModel):
    text: str


# Определяем модель данных для выходных параметров
class OutputData(BaseModel):
    prediction: str


app = FastAPI()
populate_db()


@app.post("/predict", response_model=OutputData)
def predict(input_data: InputData):
    # Получаем входные параметры
    client_request = input_data.text

    prediction = answer(client_request)

    # Возвращаем предсказание в виде объекта OutputData
    return OutputData(prediction=prediction)
# 'За месяц, предшествующий месяцу обращения