import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

dataset_path ='opsd_germany_daily.csv'

# Read data from .csv file
opsd_daily = pd.read_csv(dataset_path, index_col=0 , parse_dates = True)
opsd_daily ['Year'] = opsd_daily . index . year
opsd_daily ['Month'] = opsd_daily . index . month
opsd_daily ['Weekday Name'] = opsd_daily . index . day_name ()

w_s = 'Wind+Solar'

sns.set(rc ={'figure.figsize':(11 , 4) })
opsd_daily['Consumption'].plot(linewidth =0.5) 
cols_plot = ['Consumption','Solar','Wind']
axes = opsd_daily[cols_plot].plot(marker ='.', alpha =0.5 , linestyle ='None',
figsize =(11 , 9) , subplots = True)
for ax in axes :
    ax.set_ylabel ('Daily Totals (GWh)')
    plt.show ()
    
fig , axes = plt. subplots (3 , 1 , figsize =(11 , 10) , sharex = True )
for name , ax in zip (['Consumption','Solar','Wind'] , axes ) :
    sns . boxplot ( data = opsd_daily , x='Month', y=name , ax=ax)
    ax. set_ylabel ('GWh')
    ax. set_title ( name )
    # Remove the automatic x- axis label from all but the bottom subplot
    if ax != axes [-1]:
        ax. set_xlabel ('')
plt.show()

# print(pd.date_range('1998-03-10','1998-03-15', freq ='D'))

times_sample = pd. to_datetime (['2013-02-03','2013-02-06','2013-02-08'])
consum_sample = opsd_daily .loc [ times_sample , ['Consumption']]. copy ()
consum_freq = consum_sample . asfreq ('D')
consum_freq ['Consumption - Forward Fill'] = consum_sample . asfreq ('D', method ='ffill')
# print(consum_freq)



data_columns = ['Consumption','Wind','Solar',w_s]

opsd_weekly_mean = opsd_daily [data_columns].resample ('W').mean()
print(opsd_weekly_mean .head(3))
print ( opsd_daily . shape [0])
print ( opsd_weekly_mean . shape [0])

# Start and end of the date range to extract
start , end ='2017 -01','2017 -06'
# Plot daily and weekly resampled time series together
fig , ax = plt.subplots ()
ax.plot ( opsd_daily .loc[ start :end ,'Solar'] ,
marker ='.', linestyle ='-', linewidth =0.5 , label ='Daily')
ax.plot ( opsd_weekly_mean . loc [ start :end ,'Solar'] ,
marker ='o', markersize =8 , linestyle ='-', label ='Weekly Mean Resample')
ax.set_ylabel ('Solar Production (GWh)')
ax.legend ()
plt.show ()

opsd_annual = opsd_daily [ data_columns ]. resample ('YE') . sum( min_count =360)
opsd_annual = opsd_annual . set_index ( opsd_annual . index . year )
opsd_annual . index . name ='Year'
# Compute the ratio of Wind + Solar to Consumption
opsd_annual ['Wind + Solar / Consumption'] = opsd_annual [w_s] / opsd_annual ['Consumption']
# print(opsd_annual . tail (3))
ax = opsd_annual .loc [2012: ,'Wind + Solar / Consumption']. plot . bar( color ='C0')
ax. set_ylabel ('Fraction')
ax. set_ylim (0 , 0.3)
ax. set_title ('Wind + Solar Share of Annual Electricity Consumption')
plt.xticks(rotation =0)
plt.show()


opsd_7d = opsd_daily [ data_columns ]. rolling (7 , center = True ) . mean ()
print(opsd_7d . head (10))  


opsd_365d = opsd_daily [ data_columns ]. rolling ( window =365 , center =True , min_periods
=360) . mean ()

# Plot daily , 7 -day rolling mean , and 365 - day rolling mean time series
fig , ax = plt . subplots ()
ax. plot ( opsd_daily ['Consumption'] , marker ='.', markersize =2 , color ='0.6',
linestyle ='None', label ='Daily')
ax. plot ( opsd_7d ['Consumption'] , linewidth =2 , label ='7 -d Rolling Mean')
ax. plot ( opsd_365d ['Consumption'] , color ='0.2', linewidth =3 ,label ='Trend (365 -d Rolling Mean )')
# Set x- ticks to yearly interval and add legend and labels
ax. xaxis . set_major_locator(mdates.YearLocator () )
ax. legend ()
ax. set_xlabel ('Year')
ax. set_ylabel ('Consumption (GWh)')
ax. set_title ('Trends in Electricity Consumption')
plt . show ()


fig , ax = plt . subplots ()
for nm in ['Wind','Solar',w_s]:
    ax. plot ( opsd_365d [nm] , label =nm)
    # Set x- ticks to yearly interval , adjust y- axis limits , add legend and labels
    ax. xaxis . set_major_locator ( mdates . YearLocator () )
    ax. set_ylim (0 , 400)
    ax. legend ()
    ax. set_ylabel ('Production (GWh)')
    ax. set_title ('Trends in Electricity Production (365 -d Rolling Means )')
plt . show ()