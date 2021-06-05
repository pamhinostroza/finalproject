# Import libraries
from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
# Load the model
model = pickle.load(open('model.pkl', 'rb'))
x_scaler = pickle.load(open('x_scaler.pkl', 'rb'))
y_scaler = pickle.load(open('y_scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == "POST" :
        train = pd.read_csv("Resources/train.csv")
        train.drop("Id", axis = 1, inplace = True)
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

        train = pd.get_dummies(train)
        x = train.drop(columns=['SalePrice'])
        y = train['SalePrice'].values.reshape(-1,1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
        x_train_final = x_train[['GrLivArea', 'OverallQual', 'YearBuilt']]
        x_test_final = x_test[['GrLivArea', 'OverallQual', 'YearBuilt']]
        x_scaler_final = StandardScaler().fit(x_train_final)
        x_train_final_scaled = x_scaler_final.transform(x_train_final)
        x_test_final_scaled = x_scaler_final.transform(x_test_final)
        square_feet = float(request.form["GrLivArea"])
        overall_qual = float(request.form["OverallQual"])
        year_built = float(request.form["YearBuilt"])
        x_predict = pd.DataFrame({'Square Footage': square_feet, 'Overall Quality':overall_qual, 'Year Built':year_built}, index = [0])
        prediction = model.predict(x_predict)
        prediction = y_scaler.inverse_transform(prediction)
    return render_template('index.html', message = f"Home Price: ${prediction}")

if __name__ == '__main__':
    app.run(port=8000, debug=True)
