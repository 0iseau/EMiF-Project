'''
Regression models for predicting effect of oil price changes on stock prices.
Models :
    - OLS Linear Regression
    - AR
    - ARMA
    - VAR
    - (HAR)

Variables :
    1 - Baseline : Stock price = f(oil price)
    2 - Macro-state interaction : Stock price = f(oil price, D_macro_state) (CFNAI)
    3 - Macro-state + shock type : Stock price = f(oil price, D_macro_state, D_shock_type)

CFNAI is monthly so our period reference will be monthly. 
Daily data will be aggregated to monthly by taking the last observation of the month.
'''
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Data preparation
data = pd.read_excel("data/raw/data_hec_project_1.xlsx", skiprows=5, sheet_name="Daily")

SP = data["SP500"].replace(["#N/A N/A", "NA", ""], pd.NA).apply(pd.to_numeric, errors="coerce")
log_returns_SP = np.log(SP).diff().dropna()


WTI = data["WTI"].replace(["#N/A N/A", "NA", ""], pd.NA).apply(pd.to_numeric, errors="coerce")
log_returns_WTI = np.log(WTI).diff().dropna()




def ols(X,y):
    '''
    OLS regression : Stock price = f(oil price)
    '''
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    return model.summary(), model.params, model.pvalues

print(ols(log_returns_WTI, log_returns_SP))