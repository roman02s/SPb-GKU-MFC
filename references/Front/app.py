import requests

import streamlit as st
from fuzzywuzzy import process


# # Список примеров похожих вопросов
# similar_questions = [
#     "Как работает модель эмбеддингов?",
#     "Что такое GPT-3 и как оно функционирует?",
#     "Какие бывают методы обработки естественного языка?",
#     "Какие библиотеки используются для создания NLP-приложений?"
# ]

# Заголовок приложения
st.title("Поиск по базе знаний")

# Ввод запроса от пользователя
query = st.text_input("Введите ваш запрос")

# # Подсказка похожих вопросов
# if query:
#     similar = process.extract(query, similar_questions, limit=3)
#     similar_questions_text = ", ".join([s[0] for s in similar if s[1] > 60])  # Выбираем те, что похожи более чем на 60%
#     if similar_questions_text:
#         st.write(f"Похожие вопросы: {similar_questions_text}")

# Кнопка для выполнения поиска
search_button = st.button("Найти")

# Подсказка для пользователя
st.markdown("Введите запрос выше, чтобы получить результаты.")

# try:
if search_button and query:
    # Отправляем запрос к вашей модели для получения результатов
    # Замените URL на адрес вашего эмбединг-сервиса
    embedding_service_url = f"http://51.250.0.86:8008/new_find_similar"
    payload = {"text": query}
    response = requests.post(embedding_service_url, json=payload)
    if response.status_code == 200:
        results = response.json()
        # results = {
        #     "prediction": "Документы, подтверждающие начисление платы за жилое помещение и коммунальные услуги, топливо и транспортные услуги для доставки этого топлива (квитанции о начислении платы за жилое помещение и коммунальные услуги, в том числе платы взноса на капитальный ремонт, платы за топливо и транспортные услуги для доставки этого топлива)\n\nДокументы, подтверждающие начисление платы за жилое помещение и коммунальные услуги, топливо и транспортные услуги для доставки этого топлива (квитанции о начислении платы за жилое помещение и коммунальные услуги, в том числе платы взноса на капитальный ремонт, платы за топливо и транспортные услуги для доставки этого топлива)"
        # }
        # print(type(results["prediction"]), len(results["prediction"]))
        results = results["prediction"].split("\n\n")
        st.write("Результаты:")
        # print("\n\n\n")
        # print("results: ", results)
        # print("\n\n\n")
        # print("LEN(RESULTS): ", len(results))
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
