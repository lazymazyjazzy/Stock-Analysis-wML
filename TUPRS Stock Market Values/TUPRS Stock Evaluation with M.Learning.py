'''
TUPRS.IS Stock Market Values Evaluation with Machine Learning Models
'''

'''
All data is scraped through Yahoo Finance and TUIK(Turkish Statistical Institute),
no raw tables are used. 

Whole data is scraped, pre-processes and used in GridSearchCV model.

Financial values for Crude Oil, currencies as USD/TRY, USD/EUR and stock prices
as well as volumes are gathered from finance.yahoo.com.

Inflation rates for Turkey is directly from Turkish governmental agency TUIK's official
website.
Note: The monthly inflations are announced within the beginning of next month. Therefore,
if the inflation value of this month is not announced; it is fulfilled from previous month 
to prevent null data.
'''

#Necessary Libraries for ML Model and Feature Pre-Processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from io import StringIO
from datetime import datetime, timedelta
import requests
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error

'''#This Yahoo scraping class is directly from StackOverFlow, user named Mike-D 
#https://stackoverflow.com/users/8040498/mike-d amazing guy.

The problem with YahooFinance is the historical data is in infinite scroll format.
Direct scraping with BS4 was not providing the whole data, this guy found a solution
without any special web scraping library.'''

class YahooFinanceHistory:
    timeout = 2
    crumb_link = 'https://finance.yahoo.com/quote/{0}/history?p={0}'
    crumble_regex = r'CrumbStore":{"crumb":"(.*?)"}'
    quote_link = 'https://query1.finance.yahoo.com/v7/finance/download/{quote}?period1={dfrom}&period2={dto}&interval=1d&events=history&crumb={crumb}'

    def __init__(self, symbol, days_back=7):
        self.symbol = symbol
        self.session = requests.Session()
        self.dt = timedelta(days=days_back)

    def get_crumb(self):
        response = self.session.get(self.crumb_link.format(self.symbol), timeout=self.timeout)
        response.raise_for_status()
        match = re.search(self.crumble_regex, response.text)
        if not match:
            raise ValueError('Could not get crumb from Yahoo Finance')
        else:
            self.crumb = match.group(1)

    def get_quote(self):
        if not hasattr(self, 'crumb') or len(self.session.cookies) == 0:
            self.get_crumb()
        now = datetime.utcnow()
        dateto = int(now.timestamp())
        datefrom = int((now - self.dt).timestamp())
        url = self.quote_link.format(quote=self.symbol, dfrom=datefrom, dto=dateto, crumb=self.crumb)
        response = self.session.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), parse_dates=['Date'])


#Creating raw dataframes for features table from scraped data.   
df_tuprs = YahooFinanceHistory('TUPRS.IS', days_back=8000).get_quote()
df_crude_oil1 = YahooFinanceHistory('CL=F', days_back=8000).get_quote()
df_usd_eur = YahooFinanceHistory('EUR=X', days_back=8000).get_quote()
df_usd_try = YahooFinanceHistory('TRY=X', days_back=8000).get_quote()

#Dropping unneccessary columns and keeping only "Closure" values for computational efficiency.
df_tuprs = df_tuprs.drop(["Open","High","Low","Adj Close"],axis=1)
df_crude_oil1 = df_crude_oil1.drop(["Open","High","Low","Adj Close"],axis=1)
df_usd_eur = df_usd_eur.drop(["Open","High","Low","Adj Close","Volume"],axis=1)
df_usd_try = df_usd_try.drop(["Open","High","Low","Adj Close","Volume"],axis=1)

df_crude_oil1.rename(columns={'Volume': 'Crud Oil Volume', "Close":"Crude Oil"}, inplace=True)
df_tuprs.rename(columns={'Volume': 'Tupras Volume', "Close":"Tupras"}, inplace=True)
df_usd_eur.rename(columns={"Close":"USD/EUR"}, inplace=True)
df_usd_try.rename(columns={"Close":"USD/TRY"}, inplace=True)

#Inner merge for whole tables.
df_merge_1 = pd.merge(df_tuprs,df_crude_oil1, how='inner',on='Date')
df_merge_2 = pd.merge(df_merge_1,df_usd_eur,how="inner",on="Date")
df = pd.merge(df_merge_2,df_usd_try,how="inner",on="Date")

#In financials after inner merge; there were around 1,2% null data due to holidays etc.
#This data is directly dropped due to low size. Could be processed but I chose not to.
df = df.dropna(axis=0, subset=['Tupras','Crude Oil',"USD/EUR","USD/TRY"])

#Inflation rates by month is gathered in a data frame.
df_tr_inf = pd.read_html("https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Istatistikler/Enflasyon+Verileri/Tuketici+Fiyatlari")[0]
df_tr_inf = df_tr_inf.drop("TÜFE (Aylık % Değişim)",axis=1)
df_tr_inf.rename(columns={"Unnamed: 0":"Date", "TÜFE (Yıllık % Değişim)":"Inf. Rate"}, inplace=True)
df_tr_inf['Date'] = pd.to_datetime(df_tr_inf['Date'], format='%m-%Y')

#Since inflation rates are given in monthly order, and I have my daily stock values; I've created a date array first.
#From date obj, and datetime library, Mon-Year created.
#With groupby.transform I filled every day within a month with the monthly inflation.
#This was a huge problem for me, took my hours to solve.
date_array = pd.date_range(start=df_tr_inf["Date"].iloc[-1], end=df["Date"].iloc[-1])
inf_dates = pd.DataFrame(date_array)
inf_dates.rename(columns={0:"Date"}, inplace=True)
df_inf = pd.merge(inf_dates,df_tr_inf,how="left",on="Date")

df_inf["Month"] = df_inf["Date"].dt.month.astype("str")
df_inf["Year"] = df_inf["Date"].dt.year.astype("str")
df_inf["Mon-Year"] = df_inf["Month"] +"-"+ df_inf["Year"]

df_inf["Inf. Rate"] = df_inf.groupby('Mon-Year')['Inf. Rate'].transform(lambda val: val.fillna(val.mean()))
df_inf = df_inf.fillna(df_tr_inf["Inf. Rate"][0])
df_inf = df_inf.drop(["Month","Year","Mon-Year"], axis=1)

#After merging inflation values within the table, necessary data created for ML with dropping the dates.
df_w_dates = pd.merge(df,df_inf,how="inner",on="Date")
df = df_w_dates.drop(["Date"],axis=1)

#X for the features and y for the labels of df will be used in model.
X = df.drop("Tupras",axis=1)
y = df["Tupras"]

'''
For those who want to use poly converter and std. scaler separately without
calling a function, they can use the data within quotes below.
'''

'''
from sklearn.preprocessing import PolynomialFeatures
polynomial_converter = PolynomialFeatures(degree=2, include_bias=False)
polynomial_converter.fit(X)
polynomial_converter.transform(X)
poly_features = polynomial_converter.transform(X)'''

'''from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(poly_features)
scaler.transform(poly_features)
scaled_poly_features = scaler.transform(poly_features)'''

# MACHINE LEARNING MODEL #

#I wanted to observe polynomial data within different columns also and to fasten computation std. scaler is used.
def convert_scale(df):
    polynomial_converter = PolynomialFeatures(degree=2, include_bias=False)
    polynomial_converter.fit(df)
    polynomial_converter.transform(df)
    poly_features = polynomial_converter.transform(df)
    scaler = StandardScaler()
    scaler.fit(poly_features)
    scaler.transform(poly_features)
    return scaler.transform(poly_features)

scaled_poly_features = convert_scale(X)

#The data was around 5k rows, 10% is selected for model testing.
X_train, X_test, y_train, y_test = train_test_split(scaled_poly_features, y, test_size=0.1, random_state=42)

#Since I'll be using GridSearchCV, ElasticNet model will be used.
base_elastic_net_model = ElasticNet(tol=1e-4,max_iter=100000)
param_grid = {"alpha":[.03,.05,0.01,0.011,0.012, 1], "l1_ratio":[.9,1]}

#This GridSearch with std. scaler takes around 20 sec, without std.scaler it is around 5 min.
#Of course it depends on the hardware.
grid_model = GridSearchCV(estimator=base_elastic_net_model,
                         param_grid=param_grid,
                         scoring="neg_mean_squared_error",
                         cv=5, verbose=2) 
grid_model.fit(X_train,y_train)
grid_model.best_params_

y_pred = grid_model.predict(X_test)

RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
MAE = mean_absolute_error(y_test,y_pred)

print(f"Root Mean Squared Error: {RMSE}, Mean Absolute Error: {MAE}")

#Root Mean Squared Error: 8.89, Mean Absolute Error: 6.91 with standard scaling.
#Root Mean Squared Error: 8.97, Mean Absolute Error: 7.04 without std. scaling took 5 min.