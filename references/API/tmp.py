from ru_rag.serve import populate_db, find_similar, answer

populate_db()

res = answer(
'Как подтвердить оплату ЖКХ при оформлении услуги "Оплата жилого помещения и коммунальных услуг детям-сиротам и детям, оставшимся без попечения родителей"?'
)
print(res)
# print(res["report"])
# print(res["report"][0])
# print(res["report"][0][1])
# print(res["report"][0][1]["row"])
