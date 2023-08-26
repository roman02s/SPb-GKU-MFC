import pandas as pd
import numpy as np

import torch.nn.functional as F
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from fastapi import FastAPI
from pydantic import BaseModel

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from ru_rag.serve import populate_db, find_similar
from ru_rag.serve import answer

# from ru_rag.utils import download_llama_model


# Определяем модель данных для входных параметров
class InputData(BaseModel):
    text: str


# Определяем модель данных для выходных параметров
class OutputData(BaseModel):
    prediction: str


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def creake_embdiings() -> Tensor:
    files_embed = ['0_100',
              '100_200',
              '200_300',
              '300_400',
              '400_500',
              '500_600',
              '600_700',
              '700_800',
              '800_900',
              '900_1000',
              '1000_1100',
              '1100_1200',
              '1200_1300',
              '1300_1400',
              '1400_1500',
              '1500_1600',
              '1600_1700',
              '1700_1800',
              '1800_1900',
              '1900_2000',
              '2000_2100',
              '2100_2200',
              '2200_2300',
              '2300_2400',
              '2400_2500',
              '2500_2600',
              '2600_end']
    passage_embeddings_list = []

    for file_embed in files_embed:
        embeds = torch.load(f"../../data/processed/hack_digit/embeddings_train_{file_embed}.pt",  map_location=torch.device('cpu'))
        print(len(embeds))
        passage_embeddings_list.append(embeds)
    passage_embeddings = torch.cat(passage_embeddings_list, dim=0)


    # Создаем пустой список для дубликатов
    duplicate_indices = []

    # Поиск дубликатов
    for i, tensor1 in enumerate(passage_embeddings_list):
        for j, tensor2 in enumerate(passage_embeddings_list):
            if i != j and torch.equal(tensor1, tensor2):
                duplicate_indices.append((i, j))

    # Вывод результатов
    if duplicate_indices:
        print("Найдены дубликаты:")
        for index_pair in duplicate_indices:
            print(f"Индексы {index_pair[0]} и {index_pair[1]}: {passage_embeddings_list[index_pair[0]]}")
    else:
        print("Дубликаты не найдены.")
    
    return passage_embeddings


tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv("/data/row/train_dataset.csv")

model.to(device)
passage_embeddings = creake_embdiings()

# populate_db()

app = FastAPI()


@app.post("/find_similar", response_model=OutputData)
def predict_find_similar(input_data: InputData):
    # Получаем входные параметры
    client_request = input_data.text

    prediction = find_similar(client_request)
    ids = []
    for i in range(len(prediction["report"])):
        ids.append(prediction["report"][i][1]["row"])
    
    # id1 = prediction["report"][0][1]["row"]
    # id2 = prediction["report"][1][1]["row"]
    
    answer = ""
    for i in range(len(prediction["report"])):
        answer += "Вопрос: " + str(df.iloc[ids[i], 0])
        answer += "\t\t"
        answer += "Ответ: " + str(df.iloc[ids[i], 1])
        answer += "\n\n"
    # Возвращаем предсказание в виде объекта OutputData
    return OutputData(prediction=answer)

# 'За месяц, предшествующий месяцу обращения




@app.post("/new_find_similar", response_model=OutputData)
def predict_new_find_similar(input_data: InputData):
    # Получаем входные параметры
    client_request = input_data.text

    query_text = f"query: {client_request}"
    
    query_batch_dict = tokenizer([query_text], max_length=512, padding=True, truncation=True, return_tensors='pt')
    query_outputs = model(**query_batch_dict.to(device))
    query_embedding = average_pool(query_outputs.last_hidden_state, query_batch_dict['attention_mask'])
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    scores = (query_embedding @ passage_embeddings.T) * 100
    highest_index = np.argmax(scores[0].cpu().detach().numpy())
    
    answer = ""
    answer += "Вопрос: " + str(df.iloc[highest_index, 0])
    answer += "\t\t"
    answer += "Ответ: " + str(df.iloc[highest_index, 1])
    answer += "\n\n"

    # Возвращаем предсказание в виде объекта OutputData
    return OutputData(prediction=answer)

# 'За месяц, предшествующий месяцу обращения




SAIGA_MODEL_NAME = "IlyaGusev/saiga2_7b_lora"
SAIGA_BASE_MODEL_PATH = "TheBloke/Llama-2-7B-fp16"
# BASE_MODEL_PATH = "meta-llama/Llama-2-7b-hf"
SAIGA_DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"
SAIGA_DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."

class Conversation:
    def __init__(
        self,
        message_template=SAIGA_DEFAULT_MESSAGE_TEMPLATE,
        system_prompt=SAIGA_DEFAULT_SYSTEM_PROMPT,
        start_token_id=1,
        bot_token_id=9225
    ):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def get_start_token_id(self):
        return self.start_token_id

    def get_bot_token_id(self):
        return self.bot_token_id

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += tokenizer.decode([self.start_token_id, self.bot_token_id])
        return final_text.strip()


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()

SAIGA_tokenizer = AutoTokenizer.from_pretrained(SAIGA_MODEL_NAME, use_fast=False)

SAIGA_config = PeftConfig.from_pretrained(SAIGA_MODEL_NAME)
SAIGA_model = AutoModelForCausalLM.from_pretrained(
    SAIGA_BASE_MODEL_PATH,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="from_pretrained",
    load_in_8bit_fp32_cpu_offload=True,
)
SAIGA_model = PeftModel.from_pretrained(
    SAIGA_model,
    SAIGA_MODEL_NAME,
    torch_dtype=torch.float16
)
model.eval()

generation_config = GenerationConfig.from_pretrained(SAIGA_MODEL_NAME)
print(generation_config)






@app.post("/new_find_similar_saiga", response_model=OutputData)
def predict_new_find_similar_saiga(input_data: InputData):
    # Получаем входные параметры
    client_request = input_data.text

    inp = f"Ответь коротко в одно предложение на запрос: {client_request}"
    conversation = Conversation()
    conversation.add_user_message(inp)
    prompt = conversation.get_prompt(SAIGA_tokenizer)

    output = generate(SAIGA_model, SAIGA_tokenizer, prompt, generation_config)


    answer = ""
    answer += "Вопрос: " + str(client_request)
    answer += "\t\t"
    answer += "Ответ: " + str(output)
    answer += "\n\n"

    # Возвращаем предсказание в виде объекта OutputData
    return OutputData(prediction=answer)

# 'За месяц, предшествующий месяцу обращения












@app.post("/predict_answer", response_model=OutputData)
def predict_answer(input_data: InputData):
    # Получаем входные параметры
    client_request = input_data.text

    prediction = answer(client_request)
    # prediction: Dict[str, str] = {
    #     "Вопрос": f"Вопрос: \n",
    #     "Ответ": f"Ответ: \n",
    #     "Источники": ["asd", "asd"]
    # }
    # Возвращаем предсказание в виде объекта OutputData
    return OutputData(prediction=str(prediction))


@app.post("/predict_answer_tfidf", response_model=OutputData)
def predict_answer_tfidf(input_data: InputData):
    # Получаем входные параметры
    client_request = input_data.text

    train_data = pd.read_excel("/data/row/train_dataset_Датасет.csv")
    train_data.dropna(inplace=True)
    train_data.reset_index(drop=True, inplace=True)
    def text_preproc(text):
        text = re.sub(r'https?://[^,\s]+,?', '', text)
        text = ' '.join(re.findall('\w+', text))
        text = re.sub(r'[0-9]+', '', text)
        return text
    train_data['QUESTION'] = train_data['QUESTION'].apply(text_preproc)
    sample_without_replacement = random.sample(list(train_data['QUESTION'].values), 10)
    text_transformer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, max_features=150000)
    X_for_search = text_transformer.fit_transform(train_data['QUESTION'].values)
    from sklearn.metrics.pairwise import cosine_similarity

    query = input_data.text
    query = text_preproc(query)
    query_vector = text_transformer.transform([query])
    cosine_similarities_with_query = cosine_similarity(query_vector, X_for_search)

    # Индексы документов, отсортированные по убыванию сходства с запросом
    similar_documents_indices = np.argsort(cosine_similarities_with_query[0])[::-1]
    result = ""
    # Вывод наиболее похожих документов
    for index in similar_documents_indices[:5]:
        result += f"Документ {index}: {train_data['QUESTION'][index]}, Сходство: {cosine_similarities_with_query[0][index]}"
        result += "\n\n"
        # Возвращаем предсказание в виде объекта OutputData
    return OutputData(prediction=str(result))



@app.post("/test_mock", response_model=OutputData)
def test_mock(input_data: InputData):
    # Получаем входные параметры
    client_request = input_data.text

    prediction: dict[str, str] = {
        "prediction": "Тизера нет\n\nВсе еще нет"
    }
    # Возвращаем предсказание в виде объекта OutputData
    return OutputData(prediction=str(prediction))
