"""
Insired by:
- https://python.langchain.com/en/latest/modules/indexes/getting_started.html#one-line-index-creation
- https://huggingface.co/spaces/IlyaGusev/saiga_13b_llamacpp_retrieval_qa/blob/main/app.py
"""

import csv
import os
import sys
from typing import List, Dict

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader

from ru_rag.custom_csv_loader import CustomCSVLoader
from ru_rag.token import Token
from ru_rag.utils import get_message_tokens, download_llama_model


CHROMADB_DIR = "chromadb"
EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MODEL_FILE_NAME = "ggml-model-q4_1.bin"
MODEL_REPO = "IlyaGusev/saiga_13b_lora_llamacpp"


def populate_db() -> None:
    global CHROMADB_DIR, EMBEDDINGS_MODEL

    text_col_name = "text" if len(sys.argv) == 1 else sys.argv[1]
    raw_docs = []
    print(text_col_name)
    data_dir = "../../data/row"
    for file_name in os.listdir(data_dir):
        if ".csv" not in file_name:
            continue
        csv_path = os.path.join(data_dir, file_name)
        loader = CSVLoader(file_path=csv_path)
        raw_docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,  # min: 50, max: 2000
        chunk_overlap=30,  # min: 0, max: 500,
    )
    Chroma.from_documents(
        text_splitter.split_documents(raw_docs),
        HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL),
        persist_directory=CHROMADB_DIR,
    ).persist()


def __find_similar(query: str) -> List[Document]:
    return Chroma(
        persist_directory=CHROMADB_DIR,
        embedding_function=HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_MODEL
        ),
    ).similarity_search(query)


def find_similar(question: str) -> Dict[str, str]:
    global CHROMADB_DIR, EMBEDDINGS_MODEL

    # if len(sys.argv) == 1:
    #     print("Пожалуйста введите запрос в кавычках")
    #     return {
    #         "error": "Пожалуйста введите запрос в кавычках",
    #     }

    # docs = __find_similar(sys.argv[1])
    docs = __find_similar(question)
    report = "\n\n".join([
        f"{doc.page_content} ({doc.metadata})"
        for doc in docs
    ])
    print(report)
    return {
        "report": report,
    }


def answer(question: str) -> Dict[str, str]:
    # if len(sys.argv) == 1:
    #     print("Пожалуйста введите запрос в кавычках")
    #     return {
    #         "error": "Пожалуйста введите запрос в кавычках",
    #     }

    # docs = __find_similar(sys.argv[1])
    docs = __find_similar(question)
    if len(docs) == 0:
        print("Ничего не найдено по вашему запросу")
        return {
            "error": "Пожалуйста введите запрос в кавычках",
        }

    model = download_llama_model(MODEL_REPO, MODEL_FILE_NAME)

    # set role
    tokens = get_message_tokens(
        model,
        Token.SYSTEM.value,
        "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.",
    )
    tokens.append(Token.LINEBREAK.value)

    # set context and query
    retrieved_docs = "\n\n".join([doc.page_content for doc in docs])
    # message = f"Контекст: {retrieved_docs}\n\nИспользуя контекст, ответь на вопрос: {sys.argv[1]}"
    message = f"Контекст: {retrieved_docs}\n\nИспользуя контекст, ответь на вопрос: {question}"
    message_tokens = get_message_tokens(model, Token.USER.value, message)
    tokens.extend(message_tokens)

    # add role tokens
    role_tokens = [model.token_bos(), Token.BOT.value, Token.LINEBREAK.value]
    tokens.extend(role_tokens)

    # summarize
    summary = ""
    for token in model.generate(tokens, temp=0.1):  # temp is between 0.0 and 2.0
        if token == model.token_eos():
            break

        summary += model.detokenize([token]).decode("utf-8", "ignore")

    # print(f"Вопрос: {sys.argv[1]}\n")
    print(f"Вопрос: {question}\n")
    print(f"Ответ: {summary}\n")
    print("Источники:")
    print("\n".join([
        f"- {doc.page_content} ({doc.metadata})"
        for doc in docs
    ]))
    return {
        "Вопрос": f"Вопрос: {question}\n",
        "Ответ": f"Ответ: {summary}\n",
        "Источники": "\n".join([
            f"- {doc.page_content} ({doc.metadata})"
            for doc in docs
        ])
    }
