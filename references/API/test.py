import requests

url = "http://51.250.0.86:8008/new_find_similar"
# data = [
#     "Каковы требования для того, чтобы воспользоваться 33-й услугой в МФЦ?",
#     "Каковы требования для получения единовременной выплаты при рождении или усыновлении ребенка в МФЦ для покупки детских товаров и еды?",
#     "Каков процесс оформления детской карты в случае рождения или усыновления нескольких детей одновременно?",
#     "Какие категории граждан не могут претендовать на получение детской карты?",
#     "Какие факторы учитываются при расчете размера единовременной компенсации при рождении ребенка?",
#     "Какой дополнительный пакет документов требуется предоставить для оформления социальной поддержки при рождении или усыновлении ребенка?",
#     "Каково официальное определение категории 'Дети, оставшиеся без попечения родителей'?",
#     "Где можно проверить информацию о назначении социальной выплаты детям-сиротам на оплату жилья и коммунальных услуг?",
#     "Когда детям-сиротам предоставляется право на социальную поддержку в виде оплаты жилья и коммунальных услуг?",
#     "Какие документы подтверждают идентичность при оформлении 583 услуги?",
#     "Где можно проверить статус выплаты по услуге оплаты жилья для детей-сирот и детей без попечения родителей?",
#     "Каким образом можно удостоверить оплату коммунальных услуг при получении социальной поддержки для детей-сирот?",
#     "По каким причинам могут отказать в принятии документов при регистрации 583 услуги?",
#     "Какие критерии необходимо учесть при оформлении услуги по ремонту жилья для детей-сирот и детей без родительского попечения?",
#     "Каков итог оказания услуги 'Денежная компенсация за путевку для детей-сирот в организации отдыха'?",
#     "Какой пакет документов требуется для представителя при обращении по услуге 'Обеспечение санаторно-курортным лечением отдельных категорий граждан'?",
#     "По каким причинам может быть отказано в оказании услуги по предоставлению социальной поддержки в виде ежемесячного пособия на ребенка-инвалида до 18 лет из семьи с инвалидами I и(или) II групп в качестве законных представителей, для покупки товаров для детей и продуктов детского питания?",
#     "Какой срок установлен для оказания услуги по предоставлению ежемесячного пособия на ребенка-инвалида до 18 лет из семьи с инвалидами I и(или) II групп в качестве законных представителей, для покупки товаров для детей, продуктов детского питания и специальных молочных продуктов?",
#     "При каких условиях пасынки и падчерицы учитываются в составе семьи при предоставлении услуги по выдаче ежемесячного пособия на ребенка-инвалида до 18 лет из семьи, где законные представители являются инвалидами I или II групп, для покупки товаров детского ассортимента и продуктов детского питания?",
#     "По каким причинам может быть отказано в оказании услуги номер 38?",
#     "Каков размер ежемесячного пособия на ребенка в возрасте от 7 до 16 лет или до завершения учебы в образовательном учреждении, предоставляющем начальное, основное или среднее образование, но не старше 18 лет?",
#     "Какой срок установлен для оказания услуги 'Пособие школьникам'?",
#     "Какой пакет документов требуется представить представителю при обращении за услугой 'Ежемесячное пособие на ребенка-инвалида'?",
#     "Какие условия и требования предъявляются к доверенности при обращении за услугой 'Выполнять отдельные функции по предоставлению дополнительных мер социальной поддержки в виде обеспечения инвалидов дополнительными техническими средствами реабилитации'?",
#     "По каким причинам гражданин может претендовать на получение услуги 'Ежемесячное пособие на ребенка в возрасте от рождения до 1,5 лет'?",
#     "Какие критерии и требования необходимо выполнить для получения услуги 'Ежемесячное пособие на ребенка с момента его рождения до 1,5 лет'?",
#     "Каков порядок оформления услуги 'Ежемесячное пособие на ребенка в возрасте от рождения до 1,5 лет' в случае рождения (или усыновления) нескольких детей одновременно?",
#     "Каковы шаги и требования для оформления услуги 'Ежемесячное пособие на ребенка в возрасте от рождения до 1,5 лет' при многоплодных родах или усыновлении нескольких детей?",
#     "Каков размер выплат в рамках услуги 'Направление средств земельного капитала в Санкт-Петербурге на покупку земельного участка для дачного строительства в пределах Российской Федерации'?"
# ]

data = [
    "Каковы требования для того, чтобы воспользоваться 33-й услугой в МФЦ? ",
    "Каковы требования для получения единовременной выплаты при рождении или усыновлении ребенка в МФЦ для покупки детских товаров и еды? ",
    "Каков процесс оформления детской карты в случае рождения или усыновления нескольких детей одновременно? ",
    "Какие категории граждан не могут претендовать на получение детской карты? ",
    "Какие факторы учитываются при расчете размера единовременной компенсации при рождении ребенка? ",
    "Какой дополнительный пакет документов требуется предоставить для оформления социальной поддержки при рождении или усыновлении ребенка? ",
    "Каково официальное определение категории 'Дети, оставшиеся без попечения родителей'? ",
    "Где можно проверить информацию о назначении социальной выплаты детям-сиротам на оплату жилья и коммунальных услуг? ",
    "Когда детям-сиротам предоставляется право на социальную поддержку в виде оплаты жилья и коммунальных услуг? ",
    "Какие документы подтверждают идентичность при оформлении 583 услуги? ",
    "Где можно проверить статус выплаты по услуге оплаты жилья для детей-сирот и детей без попечения родителей? ",
    "Каким образом можно удостоверить оплату коммунальных услуг при получении социальной поддержки для детей-сирот? ",
    "По каким причинам могут отказать в принятии документов при регистрации 583 услуги? ",
    "Какие критерии необходимо учесть при оформлении услуги по ремонту жилья для детей-сирот и детей без родительского попечения? ",
    "Каков итог оказания услуги 'Денежная компенсация за путевку для детей-сирот в организации отдыха'? ",
    "Какой пакет документов требуется для представителя при обращении по услуге 'Обеспечение санаторно-курортным лечением отдельных категорий граждан'? ",
    "По каким причинам может быть отказано в оказании услуги по предоставлению социальной поддержки в виде ежемесячного пособия на ребенка-инвалида до 18 лет из семьи с инвалидами I и(или) II групп в качестве законных представителей, для покупки товаров для детей и продуктов детского питания? ",
    "Какой срок установлен для оказания услуги по предоставлению ежемесячного пособия на ребенка-инвалида до 18 лет из семьи с инвалидами I и(или) II групп в качестве законных представителей, для покупки товаров для детей, продуктов детского питания и специальных молочных продуктов? ",
    "При каких условиях пасынки и падчерицы учитываются в составе семьи при предоставлении услуги по выдаче ежемесячного пособия на ребенка-инвалида до 18 лет из семьи, где законные представители являются инвалидами I или II групп, для покупки товаров детского ассортимента и продуктов детского питания? ",
    "По каким причинам может быть отказано в оказании услуги номер 38? ",
    "Каков размер ежемесячного пособия на ребенка в возрасте от 7 до 16 лет или до завершения учебы в образовательном учреждении, предоставляющем начальное, основное или среднее образование, но не старше 18 лет? ",
    "Какой срок установлен для оказания услуги 'Пособие школьникам'? ",
    "Какой пакет документов требуется представить представителю при обращении за услугой 'Ежемесячное пособие на ребенка-инвалида'? ",
    "Какие условия и требования предъявляются к доверенности при обращении за услугой 'Выполнять отдельные функции по предоставлению дополнительных мер социальной поддержки в виде обеспечения инвалидов дополнительными техническими средствами реабилитации'? ",
    "По каким причинам гражданин может претендовать на получение услуги 'Ежемесячное пособие на ребенка в возрасте от рождения до 1,5 лет'? ",
    "Какие критерии и требования необходимо выполнить для получения услуги 'Ежемесячное пособие на ребенка с момента его рождения до 1,5 лет'? ",
    "Каков порядок оформления услуги 'Ежемесячное пособие на ребенка в возрасте от рождения до 1,5 лет' в случае рождения (или усыновления) нескольких детей одновременно? ",
    "Каковы шаги и требования для оформления услуги 'Ежемесячное пособие на ребенка в возрасте от рождения до 1,5 лет' при многоплодных родах или усыновлении нескольких детей? ",
    "Каков размер выплат в рамках услуги 'Направление средств земельного капитала в Санкт-Петербурге на покупку земельного участка для дачного строительства в пределах Российской Федерации'?"
]

import re

for i in range(len(data)):
	response = requests.post(url, json={"text": data[i]})
	result = response.json()
	prediction = result.get("prediction")

	# Используем регулярное выражение для поиска ответа
	match = re.search(r"Ответ: (.+)", prediction)

	if match:
		answer = match.group(1)
		print(f"{answer}")

	# print(f"Prediction: {prediction}")
