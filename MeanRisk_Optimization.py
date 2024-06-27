# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 04:32:52 2024

@author: Oscar Paulse
"""
import pandas as pd
import yfinance as yt
from plotly.io import show, write_image
import os
import kaleido
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.optimization import EqualWeighted, MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
from sklearn.pipeline import Pipeline
from skfolio.pre_selection import DropCorrelated
from sklearn import set_config
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    cross_val_predict,
    optimal_folds_number
    )

tickers = ['ARI.JO', 'APN.JO','EXX.JO','HAR.JO','MPT.JO','MTN.JO','OCE.JO','RBX.JO','THA.JO', 'WBO.JO']

start_date = '2022-01-01'
end_date = '2024-06-25'
df = yt.download(tickers,start = start_date, end = end_date)['Adj Close']
returns = prices_to_returns(df)
X_train, X_test = train_test_split(returns, test_size=0.3, shuffle=False)

# Minimum CVaR model fitted to training data

model = MeanRisk(
        risk_measure = RiskMeasure.CVAR,
        objective_function = ObjectiveFunction.MINIMIZE_RISK,
        portfolio_params = dict(name="Min CVaR")
    )
model.fit(X_train)
model.weights_
weights = pd.DataFrame()
weights.index = tickers
weights['weights'] = model.weights_.round(2)*5000
weights = weights.sort_values('weights', ascending= False)

benchmark = EqualWeighted(portfolio_params=dict(name="Equal Weighted"))
benchmark.fit(X_train)

# Removing Higly Correlated Stocks
set_config(transform_output = "pandas")
model2 = Pipeline([("pre_selection", DropCorrelated(threshold=0.5)),
                  ("optimization", MeanRisk(risk_measure=RiskMeasure.CVAR, 
                                            objective_function = ObjectiveFunction.MINIMIZE_RISK,
                                            portfolio_params = dict(name="Min CVaR")))])
model2.fit(X_train)
features_selected = model2['pre_selection'].get_support(indices=False)*1
def mask_extract(labels, mask):
    return [s for s,m in zip(labels,mask) if m==1]

weights2 = pd.DataFrame()
weights2.index = mask_extract(tickers,features_selected)
weights2['weights'] = model2["optimization"].weights_.round(2)*5000
weights2 = weights2.sort_values('weights', ascending= False)


pred_model = model.predict(X_test)
pred_model2 = model2.predict(X_test)
pred_bench = benchmark.predict(X_test)

print(pred_model.cvar)
print(pred_model2.cvar)
print(pred_bench.cvar)

population = Population([pred_model,pred_model2, pred_bench])
summary = population.summary()
#population.plot_composition()

fig = population.plot_cumulative_returns()

"""
if not os.path.exists("images"):
    os.mkdir("images")

import plotly.io as pio
pio.orca.config.use_xvfb = True
"""

show(fig)#.write_image("images/fig1.png")































