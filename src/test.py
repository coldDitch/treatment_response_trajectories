from preprocess import public_data

data = public_data(ids=list(range(14)))

for id in range(14):
    print(id)
    print(sum(data['df_gluc']['id'] == id))


