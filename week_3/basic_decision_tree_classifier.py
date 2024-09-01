from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


iris_X, iris_y = datasets.load_iris(return_X_y=True)

X_train , X_test , y_train , y_test = train_test_split (
                                        iris_X , iris_y ,
                                        test_size =0.2 ,
                                        random_state =42)


dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train )

y_pred = dt_classifier . predict ( X_test )
accuracy = accuracy_score ( y_test , y_pred )


print(accuracy)