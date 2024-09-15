import numpy as np
import pandas as pd
import matplotlib . pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , mean_squared_error

dataset_path = 'week_4\\Housing.csv'
df = pd.read_csv(dataset_path)
# print(df.head(5))


categorical_cols = df.select_dtypes(include =['object']).columns.to_list()
# print(categorical_cols)

ordinal_encoder = OrdinalEncoder()
encoded_categorical_cols = ordinal_encoder.fit_transform(df[ categorical_cols])
encoded_categorical_df = pd.DataFrame(
                            encoded_categorical_cols ,
                            columns = categorical_cols
                            )
numerical_df = df.drop(categorical_cols, axis=1)
encoded_df = pd.concat([numerical_df , encoded_categorical_df] , axis=1)

normalizer = StandardScaler ()
dataset_arr = normalizer.fit_transform(encoded_df)


X, y = dataset_arr[: ,1:] , dataset_arr[: , 0]

test_size = 0.3
random_state = 1
is_shuffle = True
X_train , X_val , y_train , y_val = train_test_split(X, y,test_size = test_size ,random_state = random_state ,shuffle = is_shuffle)


# Random Forest Regressor
RF_regressor = RandomForestRegressor(n_estimators=100 ,random_state = random_state)
RF_regressor.fit( X_train , y_train )


# AdaBoost Regressor
ADB_regressor = AdaBoostRegressor(n_estimators=100,random_state = random_state)
ADB_regressor.fit( X_train , y_train )

# GradientBoost Regressor
GDB_regressor = GradientBoostingRegressor (n_estimators=100,random_state = random_state)
GDB_regressor.fit( X_train , y_train )

# Predict
RF_y_pred = RF_regressor.predict(X_val)

ADB_y_pred = ADB_regressor.predict(X_val)

GDB_y_pred = GDB_regressor.predict(X_val)


# Validate with Random Forest
mae_1 = mean_absolute_error(y_val , RF_y_pred )
mse_1 = mean_squared_error(y_val , RF_y_pred )

print ('Evaluation results on validation set with Random Forest:')
print (f'Mean Absolute Error : {mae_1}')
print (f'Mean Squared Error : {mse_1}')

# Validate with AdaBoost
mae_2 = mean_absolute_error(y_val , ADB_y_pred )
mse_2 = mean_squared_error(y_val , ADB_y_pred )

print ('Evaluation results on validation set with AdaBoost:')
print (f'Mean Absolute Error : {mae_2}')
print (f'Mean Squared Error : {mae_2}')


# Validate with GradientBoost
mae_3 = mean_absolute_error(y_val , GDB_y_pred )
mse_3 = mean_squared_error(y_val , GDB_y_pred )

print ('Evaluation results on validation set with GradientBoost:')
print (f'Mean Absolute Error : { mae_3}')
print (f'Mean Squared Error : { mse_3}')