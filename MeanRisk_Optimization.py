# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 04:32:52 2024

@author: Oscar Paulse
"""
import numpy as np
import pandas as pd
import yfinance as yt
from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.optimization import EqualWeighted, MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns

tickers = ['ARI.JO', 'APN.JO','EXX.JO','HAR.JO','MPT.JO','MTN.JO','OCE.JO','RBX.JO','THA.JO', 'WBO.JO']

start_date = '2014-01-01'
end_date = '2024-06-25'
df = yt.download(tickers,start = start_date, end = end_date)['Adj Close']
#returns = df.pct_change().dropna()
returns = prices_to_returns(df)
X_train, X_test = train_test_split(returns, test_size=0.1, shuffle=False)

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
benchmark.weights_


pred_model = model.predict(X_test)
pred_bench = benchmark.predict(X_test)

print(pred_model.cvar)
print(pred_bench.cvar)

population = Population([pred_model, pred_bench])
population.plot_composition()


fig = population.plot_cumulative_returns()
show(fig)

summary = population.summary()





























