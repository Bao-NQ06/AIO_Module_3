import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


data = {'age': [23, 25, 27, 29, 29],
        'likes english': [0, 1, 1, 0, 0],
        'likes ai': [0, 1, 0, 1, 0],
        'raise salary': [0, 0, 1, 1, 0]}
df = pd.DataFrame(data)
X = df[['age', 'likes english', 'likes ai']].values
y = df[['raise salary']].values.reshape(-1,)

# clf = DecisionTreeClassifier()
clf = DecisionTreeClassifier(criterion='entropy')


clf = clf.fit(X, y)



#Test case

x_test = np.array([[27, 0, 1]])
predicted_label = clf.predict(x_test)
print(f'Predict: {predicted_label}')


plot_tree(clf, feature_names=['age', 'likes english', 'likes ai'], fontsize=10)
plt.show()


# Test with Iris dataset
iris_X, iris_y = datasets.load_iris(return_X_y=True)

X_train , X_test , y_train , y_test = train_test_split (
                                        iris_X , iris_y ,
                                        test_size =0.2 ,
                                        random_state =42)


dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train )

y_pred = dt_classifier . predict ( X_test )
accuracy = accuracy_score ( y_test , y_pred )


plot_tree(dt_classifier, feature_names=["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"])

plt.show()