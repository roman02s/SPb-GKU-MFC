import requests

import streamlit as st
from fuzzywuzzy import process

IP = "51.250.0.86"
PORT = 8000
# Заголовок приложения
st.title("Поиск по базе знаний")

# Ввод запроса от пользователя
query = st.text_input("Введите ваш запрос")
# Кнопка для выполнения поиска
search_button = st.button("Найти похожие")

send_button = st.button("Отправить запрос в Сайгу")

# Подсказка для пользователя
st.markdown("Введите запрос выше, чтобы получить результаты.")

# try:
if search_button and query:
    # Отправляем запрос к вашей модели для получения результатов
    # Замените URL на адрес вашего эмбединг-сервиса
    embedding_service_url = f"http://{IP}:{PORT}/new_find_similar"
    payload = {"text": query}
    response = requests.post(embedding_service_url, json=payload)
    if response.status_code == 200:
        results = response.json()
        results = results["prediction"].split("\n\n")
        st.write("Результаты:")
        try:
            for idx, result in enumerate(results[:-1], start=1):
                # print("="*100)
                print("result: ", result)
                question, answer = result.split("\t\t")
                print("question, answer: ", question, answer)
                st.write(f"{idx:5}. {question}\n")
                st.write(f"{' ':6} {answer}")
        except BaseException as err:
            print("Error: ", err)
    else:
        st.write("Произошла ошибка при получении результатов")
# except BaseException as err:
#     print("Error: ", err)


if send_button and query:
    embedding_service_url = f"http://{IP}:{PORT}/new_find_similar_saiga"
    payload = {"text": query}
    response = requests.post(embedding_service_url, json=payload)
    if response.status_code == 200:
        results = response.json()
        results = results["prediction"].split("\n\n")
        st.write("Результаты:")
        try:
            for idx, result in enumerate(results[:-1], start=1):
                print("="*100)
                print("result: ", result)
                question, answer = result.split("\t\t")
                print("question, answer: ", question, answer)
                st.write(f"{idx:5}. {question}\n")
                st.write(f"{' ':6} {answer}")
        except BaseException as err:
            print("Error: ", err)
    else:
        st.write("Произошла ошибка при получении результатов")