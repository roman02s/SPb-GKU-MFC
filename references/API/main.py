import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from ru_rag.serve import answer, populate_db, find_similar
from ru_rag.utils import download_llama_model


# Определяем модель данных для входных параметров
class InputData(BaseModel):
    text: str


# Определяем модель данных для выходных параметров
class OutputData(BaseModel):
    prediction: str


populate_db()
model = download_llama_model(MODEL_REPO, MODEL_FILE_NAME)
app = FastAPI()


@app.post("/find_similar", response_model=OutputData)
def predict_find_similar(input_data: InputData):
    # Получаем входные параметры
    client_request = input_data.text

    prediction = find_similar(client_request)

    id1 = prediction["report"][0][1]["row"]
    id2 = prediction["report"][1][1]["row"]
    df = pd.read_csv("/data/row/train_dataset.csv")

    answer = df.iloc[id1, 1]
    answer += str("\n\n") + str(df.iloc[id2, 1])

    # Возвращаем предсказание в виде объекта OutputData
    return OutputData(prediction=answer)
    
# 'За месяц, предшествующий месяцу обращения

@app.post("/predict_answer", response_model=OutputData)
def predict_answer(input_data: InputData):
    # Получаем входные параметры
    client_request = input_data.text

    prediction = answer(model, client_request)
    # prediction: Dict[str, str] = {
    #     "Вопрос": f"Вопрос: \n",
    #     "Ответ": f"Ответ: \n",
    #     "Источники": ["asd", "asd"]
    # }
    # Возвращаем предсказание в виде объекта OutputData
    return OutputData(prediction=str(prediction))