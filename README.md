# TUPRS.IS Stock Analysis w/SciKit-Learn 
## Simple ML Model for Predicting TUPRS.IS Stock Prices

[TUPRS-IS](https://finance.yahoo.com/quote/TUPRS.IS?p=TUPRS.IS&.tsrc=fin-srch) is a stock that you can buy globally, belongs to TUPRAS Turkiye Petroleum Refineries which is one of the biggest companies in Turkey with the market cap around 24,5 B.

<img width="933" alt="Ekran Resmi 2021-05-14 13 56 41" src="https://user-images.githubusercontent.com/82866518/118261435-37ec0c80-b4bc-11eb-8991-013df94e4715.png">

This machine learning model is created for observing the effects of following criterias on TUPRS.IS stock market price;

* TUPRS.IS Stock Market Volume
* [Crude Oil Prices in USD](https://finance.yahoo.com/quote/CL=F?p=CL=F&.tsrc=fin-srch)
* Crude Oil Volume
* [USD/EUR Currency Rate](https://finance.yahoo.com/quote/EUR=X?p=EUR=X&.tsrc=fin-srch)
* [USD/TRY Currency Rate](https://finance.yahoo.com/quote/TRY=X?p=TRY=X&.tsrc=fin-srch) 
* [Turkish Monthly Inflation Rates](https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Statistics/Inflation+Data)

Since it is a Turkish company, USD/TRY currency is also evaluated along with the inflation rates.

All data provided for the model is scraped through web, and features are pre-processed within the script I wrote.
You can directly copy the script and run if you have necessary libraries.

For Yahoo Finance scraping, I've used a class written in [Stack Overflow by Mike-D](https://stackoverflow.com/questions/44225771/scraping-historical-data-from-yahoo-finance-with-python). Yahoo Finance provides the historical data in infinite scrolling format. It is really hard to scrape through with classical BS4 and lxml.

<img width="525" alt="Ekran Resmi 2021-05-14 14 19 58" src="https://user-images.githubusercontent.com/82866518/118263847-79ca8200-b4bf-11eb-98d0-f8388da7e22d.png">

Versions for the libraries I've used; 
jupyter==1.0.0, lxml==4.5.1, MarkupSafe==1.1.1, matplotlib==3.3.2, notebook==6.0.3, numpy==1.18.1, openpyxl==3.0.4, pandas==1.1.2, Pillow==7.2.0, scikit-learn==0.23.2, scipy==1.4.1, seaborn==0.11.0, SQLAlchemy==1.3.18.

<img width="504" alt="Ekran Resmi 2021-05-14 14 18 43" src="https://user-images.githubusercontent.com/82866518/118263739-56073c00-b4bf-11eb-96fe-aa0cfc2026d5.png">

## Quick Look On Real World Data Frame and Graphs

This was the head(first few lines) of core dataframe I've used after scraping to provide graphs etc.
There were around 5k rows, having the data from 2005 to today.

<img width="834" alt="Ekran Resmi 2021-05-14 14 07 37" src="https://user-images.githubusercontent.com/82866518/118262586-c0b77800-b4bd-11eb-895a-752f24b5adb6.png">

### Let's check some real data in scatter plots before we dive into ML model and predictions.
Note that some values are normalized in order for clear reading.

#### Norm. Tupras vs USD/EUR and USD/TRY Currency
![Norm  Tupras vs USD](https://user-images.githubusercontent.com/82866518/118263966-aaaab700-b4bf-11eb-9024-8f07edd5a1d0.png)

### Tupras vs Crude Oil Prices in TRY
![Tupras vs Crud Oil TRY](https://user-images.githubusercontent.com/82866518/118264047-c910b280-b4bf-11eb-8d86-7049b8820456.png)

### Tupras vs Crude Oil Prices in USD
![Tupras vs Crud Oil USD](https://user-images.githubusercontent.com/82866518/118264057-cd3cd000-b4bf-11eb-8495-a7cf6edb015f.png)

### Norm. Tupras vs TRY Inflation Rates
![Tupras vs TRY Inf  Rate](https://user-images.githubusercontent.com/82866518/118264067-d037c080-b4bf-11eb-8b65-1d99838577cd.png)


## Machine Learning Model and Predictions for Different Variables

Since the data frame is really complicated, and as you know stock values actually depends on millions of stuff. I've chosen to use poly features in my model. 
Also after using poly features, values are scaled using StandardScaler for fast computation(almost 20 times faster).

<img width="543" alt="Ekran Resmi 2021-05-14 14 37 42" src="https://user-images.githubusercontent.com/82866518/118265503-f2cad900-b4c1-11eb-8c19-47970e91c610.png">

### Elastic Net Regularization
![image](https://user-images.githubusercontent.com/82866518/118266470-5275b400-b4c3-11eb-8a78-e8e6e3da476a.png)
With train_test_split, 10% of the data is chosen for final test and models were created to find best "alpha" for Lambda in the ElasticNet equation above and "l1_ratio" which is the alpha which arranges the weight of LASSO and Ridge Regressions.

<img width="681" alt="Ekran Resmi 2021-05-14 14 39 39" src="https://user-images.githubusercontent.com/82866518/118265681-3887a180-b4c2-11eb-8f67-510a82c724d1.png">

After fitting the model to training set, RMSE(Root Mean Squared Error) and MAE(Mean Absolute Error) are calculated accordingly.
<img width="366" alt="Ekran Resmi 2021-05-14 14 52 18" src="https://user-images.githubusercontent.com/82866518/118266939-fcedd700-b4c3-11eb-8453-0a02f6f9160e.png">

Values were not suprising for the model; ElasticNet(alpha=0.01, l1_ratio=1, max_iter=100000). Basically saying LASSO Regression was way better than using Ridge.

The final results with std. scaler + poly_features;
#Root Mean Squared Error: 8.89, Mean Absolute Error: 6.91 with standard scaling.
#Root Mean Squared Error: 8.97, Mean Absolute Error: 7.04 without std. scaling.

### In short; our ML model is working around 7-10% error for predicting TUPRS.IS stock prices depending on other parameters. BTW, we should definitely not put any money based on this model which has 7-10% error. It would be really naive.

As a quick note; I've also tried ElasticNetCV, LASSO and Ridge Regressions on the dataframe with different scaling options. RMSE.mean() were around 25%, way too high; so I've continued with GridSearchCV + ElasticNet which is around 10%, seemed nice.


## Effects of Other Variables on Stock Prices

We created our model with an acceptable error(accepted by me), let's find out what variables are affecting the stock prices.
These graphs are created by keeping all parameters stable and only changing one in order to observe clear effects.

__This is where we will have really interesting information about the effects of other parameters on stock prices.__

### Crude Oil Price [USD]
Interesting observation; it has a peak and optimum value around 75 USD but converging down for too high and too low.
<img width="499" alt="Crud Oil Price" src="https://user-images.githubusercontent.com/82866518/118268105-a08bb700-b4c5-11eb-8a78-7eb6ef38c98b.png">

### TUPRS.IS Stock and Crude Oil Volume in Stock Market
Both volume increases have linear effect on stock prices. For Tüpraş volume, it is easy to guess; if it's bought more, it will increase.
However, for the crude oil volume it didn't make sense to me. Maybe someone who knows can explain better.
<img width="473" alt="Crud Oil Vol" src="https://user-images.githubusercontent.com/82866518/118268117-a4b7d480-b4c5-11eb-8b66-cd7848dcc0b5.png">
<img width="503" alt="Tupras Vol" src="https://user-images.githubusercontent.com/82866518/118268130-aa151f00-b4c5-11eb-978c-02d4a72ddaa4.png">

### TRY Inflation Rate
Inflation rate has positive effect on stock prices. However, as you may guess inflation also means value loss on currency(in this case TRY). So this parameter actually should be deeply analysed by someone who can understand finance.
<img width="477" alt="Inf Rate" src="https://user-images.githubusercontent.com/82866518/118268123-a71a2e80-b4c5-11eb-8837-01c90e98032f.png">

### USD/TRY Currency 
This is where it gets really interesting. TUPRS.IS stock values actually really dependent on USD/TRY currency and according to our model it has an optimum value around 1 USD = 5.5 TRY. __If USD/TRY goes above 11 TRY, TUPRS.IS stocks losing all of its value and becomes 0 TRY directly. (maybe not possible in real world, but can say something for the future if USD/TRY goes up like that.)__
<img width="493" alt="USD:TRY" src="https://user-images.githubusercontent.com/82866518/118268144-b00b0000-b4c5-11eb-9323-cb3422aa243d.png">

### USD/EUR Currency
This graph actually explains what happens if USD loses or earns money compared to EUR. According to our ML model, it has the lowest stock price value around 0.7.
If USD earns more value and goes up to 0.9, TUPRS.IS also increases it's price. But same situation also goes on for the complete opposite scenario. I didn't find this graph meaningful, but added anyways. 
<img width="507" alt="USD:EUR" src="https://user-images.githubusercontent.com/82866518/118268135-ac777900-b4c5-11eb-841f-93eea051528c.png">

In conclusion, stock prices depends on millions of parameters. Maybe it can be modeled up to some degree, but in real world it is almost impossible to come up with a future value just like that.

In this sample I've tried to analyse one of the biggest company of Turkey, without it's financials and find some useful data as well as useless ones.

If you came this far, thank you for your interest.
Have a great day ahead!
