import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from six import StringIO
import pydotplus
from IPython.display import Image 

df = pd.read_csv('dataset_einstein.csv', delimiter=';')

# Get the size of the dataframe
count_row = df.shape[0]
count_col = df.shape[1]

# Clean fields with no information
df = df.dropna()

# Convert dataframe to python's array
Y = df['SARS-Cov-2 exam result'].values
X = df[['Hemoglobin', 'Leukocytes', 'Basophils', 'Proteina C reativa mg/dL']].values

# Divide dataset between train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

tree_alg = DecisionTreeClassifier(criterion='entropy', max_depth=5)
model = tree_alg.fit(X_train, Y_train)

name_features = ['Hemoglobin', 'Leukocytes', 'Basophils','Proteina C reativa mg/dL']
name_classes = model.classes_

""" dot_data = StringIO()
export_graphviz(model, out_file=dot_data, filled=True, feature_names=name_features, class_names=name_classes, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png)
graph.write_png('tree.png')
Image('tree.png') """

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
print('Feature ranking:')

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
f, ax = plt.subplots(figsize=(11, 9))
plt.title("Feature ranking", fontsize = 20)
plt.bar(range(X.shape[1]), importances[indices],
    color="b", 
    align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.ylabel("importance", fontsize = 18)
plt.xlabel("index of the feature", fontsize = 18)
plt.show()

predict_Y = model.predict(X_test)
print('Tree Accuracy:', accuracy_score(Y_test, predict_Y))
print(classification_report(Y_test, predict_Y))