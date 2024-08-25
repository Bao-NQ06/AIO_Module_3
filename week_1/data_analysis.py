import numpy as np
import pandas as pd
import matplotlib . pyplot as plt

dataset_path = 'IMDB-Movie-Data.csv'

# Read data from .csv file
data = pd.read_csv( dataset_path )


data_indexed = pd.read_csv(dataset_path , index_col ="Title")

genre = data [['Genre']]
revenue =  data_indexed['Revenue (Millions)']


some_cols = data [['Title','Genre','Actors','Director','Rating']]
# print(
# data [(( data ['Year'] >= 2010) & ( data ['Year'] <= 2015) )
#     & ( data ['Rating'] < 6.0)
#     & ( data ['Revenue (Millions)'] > data ['Revenue (Millions)']. quantile (0.95) ) ]
# )

# print(data.groupby('Director')[['Rating']].mean(). sort_values ([ 'Rating'] , ascending =False ) . head (10))

# print(data . drop ('Metascore', axis =1) . head ())

revenue_mean = revenue.mean ()
print ("The mean revenue is: ", revenue_mean )

# We can fill the null values with this mean revenue
data_indexed['Revenue (Millions)']= revenue.fillna(revenue_mean)


def rating_group ( rating ) :
    if rating >= 7.5:
        return 'Good'
    elif rating >= 6.0:
        return 'Average'
    else :
        return 'Bad'
    
data ['Rating_category'] = data['Rating'].apply(rating_group)
print(data [[ 'Title','Director','Rating','Rating_category']]. head (5))