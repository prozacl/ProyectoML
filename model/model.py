from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pickle

url = 'https://raw.githubusercontent.com/seba000/csv_csgo_entrega2/main/Anexo%20Forma%20B_demo_round_traces.csv'

X, y = url(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

clf = RandomForestClassifier()
print(clf.fit(X_train, y_train).score(X_test, y_test))

filename = 'checkpoints/model.pkl'
pickle.dump(clf, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
