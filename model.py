# Loading Data and Packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

# Reading CVSs
train = pd.read_csv("Resources/train.csv")

# Dropping 'Id' column
train.drop("Id", axis = 1, inplace = True)

# Cleaning up data (string columns with None and integer columns with 0)
train["PoolQC"] = train["PoolQC"].fillna("None")
train["MiscFeature"] = train["MiscFeature"].fillna("None")
train["Alley"] = train["Alley"].fillna("None")
train["Fence"] = train["Fence"].fillna("None")
train["FireplaceQu"] = train["FireplaceQu"].fillna("None")
train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    train[col] = train[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    train[col] = train[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    train[col] = train[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train[col] = train[col].fillna('None')
train["MasVnrType"] = train["MasVnrType"].fillna("None")
train["MasVnrArea"] = train["MasVnrArea"].fillna(0)
train['MSZoning'] = train['MSZoning'].fillna(train['MSZoning'].mode()[0])
train = train.drop(['Utilities'], axis=1)
train["Functional"] = train["Functional"].fillna("Typ")
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])
train['KitchenQual'] = train['KitchenQual'].fillna(train['KitchenQual'].mode()[0])
train['Exterior1st'] = train['Exterior1st'].fillna(train['Exterior1st'].mode()[0])
train['Exterior2nd'] = train['Exterior2nd'].fillna(train['Exterior2nd'].mode()[0])
train['SaleType'] = train['SaleType'].fillna(train['SaleType'].mode()[0])
train['MSSubClass'] = train['MSSubClass'].fillna("None")

# Getting dummies for string based columns
train = pd.get_dummies(train)

# Setting up x(target) and y(data) values
x = train.drop(columns=['SalePrice'])
y = train['SalePrice'].values.reshape(-1,1)

# Splitting data into test and train
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

# Creating scaler
x_scaler = StandardScaler().fit(x_train)
y_scaler = StandardScaler().fit(y_train)

# Transforming with scaler
x_train_scaled = x_scaler.transform(x_train)
x_test_scaled = x_scaler.transform(x_test)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# Creating empty regular linear regression model
regressor = LinearRegression()

# Training model
regressor.fit(x_train_scaled, y_train_scaled)

# Making predictions on test portion
reg_predictions = regressor.predict(x_test_scaled)
reg_MSE = mean_squared_error(y_test_scaled, reg_predictions)
r2_reg = regressor.score(x_test_scaled, y_test_scaled)

# Training lasso linear regression model
lasso = Lasso(alpha=.01).fit(x_train_scaled, y_train_scaled)

# Making predictions on test portion
predictions = lasso.predict(x_test_scaled)
MSE = mean_squared_error(y_test_scaled, predictions)
r2_las = lasso.score(x_test_scaled, y_test_scaled)

# Creating empty random forest regression model
rf = RandomForestRegressor(n_estimators=200)

# Training random forest regression model
rf = rf.fit(x_train, y_train)

# Making predictions on test portion
rf_predictions = rf.predict(x_test_scaled)
rf_MSE = mean_squared_error(y_test_scaled, rf_predictions)
r2_rf = rf.score(x_test, y_test)

# Training ridge linear regression model
ridge = Ridge(alpha=.01).fit(x_train_scaled, y_train_scaled)

# Making predictions on test portion
ridge_predictions = ridge.predict(x_test_scaled)
ridge_MSE = mean_squared_error(y_test_scaled, ridge_predictions)
r2_ridge = ridge.score(x_test_scaled, y_test_scaled)

# Training elastic net linear regression model
en = ElasticNet(alpha=.01).fit(x_train_scaled, y_train_scaled)

# Making predictions on test portion
en_predictions = en.predict(x_test_scaled)
en_MSE = mean_squared_error(y_test_scaled, en_predictions)
r2_en = en.score(x_test_scaled, y_test_scaled)

# Lasso is slightly better than Elastic Net so it will be chosen to continue


# Inversing scaling on y
predictions = y_scaler.inverse_transform(predictions)

# Reducing y_test from list of lists to just a list
y_test_ravel = np.ravel(y_test)

# Turning raveled predictions and y_test_scaled into a dataframe to evaluate
pd.DataFrame({"predictions": predictions,"true values": y_test_ravel})

# Listing variables and their scores in descending order
lasso_scores = sorted(list(zip(x_train, lasso.coef_)),reverse=True)

# Turning variables and their scores into a dataframe to evaluate
var_df = pd.DataFrame(lasso_scores)

# Renaming columns
var_df.columns=["variables","coefficients"]

# Taking absolute value of coefficients
var_df['coefficients'] = abs(var_df['coefficients'])

# Sorting absolute value of coefficients
lasso_sorted = var_df.sort_values(by="coefficients",ascending=False)
top_10 = lasso_sorted.head(10)


# Top 10 explained
# GrLivArea: living area square feet (above ground)
# RoofMatl_ClyTile: Clay or Tile roof material
# OverallQual: Rates the overall material and finish of the house
# Condition2_PosN: Near to positive off-site feature (park, greenbelt, etc.)
# BsmtQual_Ex: height of the basement (100+ inches)
# YearBuilt: Original construction date
# TotalBsmtSF: Total square feet of basement area
# BsmtFinSF1: Rating of basement finished area's squared feet
# KitchenQual_Ex: Excellent Kitchen quality
# Neighborhood_NoRidge: Northridge neighborhood


# Resplitting from top 3 variables only
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
x_train_final = x_train[['GrLivArea', 'OverallQual', 'YearBuilt']]
x_test_final = x_test[['GrLivArea', 'OverallQual', 'YearBuilt']]

# Creating new scaler
x_scaler_final = StandardScaler().fit(x_train_final)

# Retransforming with scaler
x_train_final_scaled = x_scaler_final.transform(x_train_final)
x_test_final_scaled = x_scaler_final.transform(x_test_final)

# Dataframing scaled x variables
x_train_2 = pd.DataFrame(x_train_scaled)
x_test_2 = pd.DataFrame(x_test_scaled)

# Retraining with lasso
lasso2 = Lasso(alpha=.01).fit(x_train_final, y_train_scaled)

# Making new predictions on top 3 variables testing portion to see difference in score
predictions2 = lasso2.predict(x_test_final)
MSE2 = mean_squared_error(y_test_scaled, predictions2)
r2_las2 = lasso2.score(x_test_final, y_test_scaled)

# Saving model and scalers to disk
pickle.dump(lasso2, open('model.pkl','wb'))
pickle.dump(x_scaler, open('x_scaler.pkl', 'wb'))
pickle.dump(y_scaler, open('y_scaler.pkl', 'wb'))

# Loading model and scalers
model = pickle.load(open('model.pkl','rb'))
x_scaler = pickle.load(open('x_scaler.pkl', 'rb'))
y_scaler = pickle.load(open('y_scaler.pkl', 'rb'))
