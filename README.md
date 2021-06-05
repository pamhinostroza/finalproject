# finalproject
1. Traditional ML problem: predict housing price based on features
 - clean data in pandas
 - pay attention to: missing data, categorical columns (one-hot encoding)
 - check out existing kaggle notebooks

2. Tableau
 - do Tableau viz (don't need flask)
     1. bar chart with feature importances (random forest activity)
     2. line chart of AVERAGE housing price by year built (kaggle data)
     
     3. SUPER-OPTIONAL: (if time series) line chart of housing price over time (zillow data)

3. OPTIONAL: HTML
 - already set up
 
4. OPTIONAL: Flask? (example provided by tutor)
 - have user input nBR, nBA, (most important features) (no database needed)
    1. jupyter notebook: train, test, save model (model.save('model.sav'))
    2. flask app: load model and make predictions based on new data (model.load('model.sav'))

5. SUPER-OPTIONAL: Time Series problem
 - getting the data
 - ARMA, ARIMA models - only need one column (data) and time column
 