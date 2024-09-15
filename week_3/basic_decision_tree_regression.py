import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt



data = {'age': [23, 25, 27, 29, 29],
        'likes english': [0, 1, 1, 0, 0],
        'likes ai': [0, 1, 0, 1, 0],
        'salary': [200, 400, 300, 500, 400]}
df = pd.DataFrame(data)
X = df[['age', 'likes english', 'likes ai']].values
y = df[['salary']].values.reshape(-1,)
reg = DecisionTreeRegressor()
reg = reg.fit(X, y)
clf = DecisionTreeClassifier()



x_test = np.array([[18, 1, 1]])
predicted_salary = reg.predict(x_test)
print(predicted_salary)

plot_tree(reg, feature_names=['age', 'likes english', 'likes ai'], fontsize=6)
plt.show()

# CPU Machine Dataset


machine_cpu = fetch_openml(name='machine_cpu')
machine_data = machine_cpu.data
machine_labels = machine_cpu.target

X_train , X_test , y_train , y_test = train_test_split(
                                    machine_data , machine_labels ,
                                    test_size =0.2 ,
                                    random_state =42)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train , y_train)


y_pred = tree_reg.predict(X_test)
mse = mean_squared_error(y_test , y_pred)

print(mse)

plot_tree(reg, feature_names=['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX'])
plt.show()