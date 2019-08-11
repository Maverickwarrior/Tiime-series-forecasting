#Environment setup
import sklearn
import scipy
import statsmodels
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pandas import Series
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import DataFrame
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.pylab import rcParams
from datetime import datetime
rcParams['figure.figsize'] = 15, 6

#reading dataset
data = pd.read_csv('raw_data.csv')

#parse strings to datetime type
data['order_date']= pd.to_datetime(data['order_date'],infer_datetime_format=True)
indexedDataset=data.set_index(['order_date'])

indexedDataset.head(5)

#dropping non necessary columns for forecasting
indexedDataset.drop(['asset_id','city_id','asset_name'],axis = 1, inplace = True)

##Plot graph
plt.xlabel("Date")
plt.ylabel("Number of orders")
plt.plot(indexedDataset)

#Determining rolling statistics
rolmean= indexedDataset.rolling(window=12).mean()
rolstd= indexedDataset.rolling(window=12).std()
print(rolmean,rolstd)

#plot rolling statistics
orig= plt.plot(indexedDataset,color='blue',label='Original')
mean= plt.plot(rolmean,color='red',label='Rolling mean')
std= plt.plot(rolstd,color='black',label='Rolling std')
plt.legend(loc='best')
plt.title('Rolling mean and Rolling std')
plt.show(block=False)


#Perform dickey-fuller test
from statsmodels.tsa.stattools import adfuller
print('Results of Dickey-Fuller Test: ')
dftest=adfuller(indexedDataset['count'],autolag='AIC')

dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key]=value
    
print(dfoutput)    
            
#estimate trend
indexedDataset_logScale= np.log(indexedDataset)
plt.plot(indexedDataset_logScale)

#Moving average
movingAverage= indexedDataset_logScale.rolling(window=12).mean()
movingSTD= indexedDataset_logScale.rolling(window=12).std()
plt.plot(indexedDataset_logScale)
plt.plot(movingAverage,color='red')

datasetLogScaleMinusMovingAverage= indexedDataset_logScale - movingAverage
datasetLogScaleMinusMovingAverage.head(12)

#remove nan values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)

#Plotting trend,seasonal,residual error
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition= seasonal_decompose(indexedDataset_logScale,freq=1)

trend= decomposition.trend
seasonal= decomposition.seasonal
residual= decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logScale,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

decomposedLogData= residual
decomposedLogData.dropna(inplace=True)



datasetLogDiffShifting= indexedDataset_logScale-indexedDataset_logScale.shift()
datasetLogDiffShifting.dropna(inplace=True)

#ACF AND PACF PLots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(datasetLogDiffShifting, nlags=20)
lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols')

#plot ACF
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


#ARIMA MODEL
from statsmodels.tsa.arima_model import ARIMA

#AR MODEL
model = ARIMA(indexedDataset_logScale, order=(2, 1,0))  
results_AR = model.fit(disp=-1)  
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-datasetLogDiffShifting)**2))

#MA model
model = ARIMA(indexedDataset_logScale, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-datasetLogDiffShifting)**2))

#ARIMA MODEL
model = ARIMA(indexedDataset_logScale, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-datasetLogDiffShifting)**2))


#original scale
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.head(5)

#cumsum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()

predictions_ARIMA_log = pd.Series(indexedDataset_logScale['count'].ix[0], index=indexedDataset_logScale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

#last step
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))

indexedDataset_logScale
results_ARIMA.plot_predict(1,5300)
x=results_ARIMA.forecast(steps=120)









