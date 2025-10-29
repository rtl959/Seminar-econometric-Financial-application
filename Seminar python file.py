##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import datetime

tickers= ["LLY", "JNJ", "ABBV", "RHHBY", "AZN", "NVS", "NVO", "MRK", "AMGN", "PFE", "GILD", "SNY", "BMY", "GSK", "BIIB", "OGN"]

start= datetime.datetime(2015, 1, 1)
end= datetime.datetime(2025, 1, 1)

def get(tickers, startdate, enddate):
    def data(ticker):
        full = pd.read_csv('C:/Users/cooki/OneDrive/Uni/Seminar Econ in financial application/Seminar kode mappe/Seminar-econometric-Financial-application/by_ticker_csv/' + tickers + '.CSV', parse_dates=[3], index_col='Date')
        full = full.loc[start:end]
        
        return full
    datas = map(data, tickers)
    full_concat = pd.concat(datas, keys=tickers, names=['Ticker', 'Date'])
    full_concat = full_concat.drop(['SNo', 'Name', 'Ticker', 'Marketcap'], axis = 1)
    full_concat['Adj Close'] = full_concat.Close

    return full_concat

def prep_data(tickers, start, end, K):
    all_data = get(tickers, startdate=start, enddate=end)
    daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')
    daily_pct_change = daily_close_px.pct_change()

    dly_vars = all_data[['Adj Close', 'Volume']].reset_index()

    dly_vars = pd.DataFrame({'Ticker': dly_vars['Ticker'],
                             'Date': dly_vars['Date'],
                             'Price': dly_vars['Adj Close'],
                             'Volume': dly_vars['Volume'],
                             't-1': 0,
                             'mkt_t-1': 0,
                             'target': 0})

    #Creating momentum variable for market

    #Market 1 . Return of market t-skip //reversal ************
    mkt_ret = daily_pct_change.mean(axis=1)
    #Market 2 . Cumulative return of market t-T to t-skip //momentum ************
    mkt_cum_3 = (mkt_ret + 1).shift(1).rolling(3).apply(np.prod) - 1
    mkt_cum_7 = (mkt_ret + 1).shift(1).rolling(7).apply(np.prod) - 1
    mkt_cum_15 = (mkt_ret + 1).shift(1).rolling(15).apply(np.prod) - 1
    mkt_cum_30 = (mkt_ret + 1).shift(1).rolling(30).apply(np.prod) - 1
    #Market 3 . standard deviation of the market from t-T to t-skip //volatility
    mkt_std_3 = (mkt_ret + 1).shift(1).rolling(3).std()
    mkt_std_7 = (mkt_ret + 1).shift(1).rolling(7).std()
    mkt_std_15 = (mkt_ret + 1).shift(1).rolling(15).std()
    mkt_std_30 = (mkt_ret + 1).shift(1).rolling(30).std()
    mkt_ret = mkt_ret.shift(1)

    #Asset specific momentum
    for ticker in tickers:
        # ticker 1 . reversal t-1
        dly_vars.loc[dly_vars.ticker == ticker, 't-1'] = dly_vars.loc[dly_vars.Ticker == ticker, 'Price'].pct_change()
        # ticker 2.1 . momentum J=3
        dly_vars.loc[dly_vars.Ticker == ticker, 'J_3'] = (dly_vars.loc[dly_vars.Ticker == ticker, 't-1'] + 1).shift(2).rolling(3 - 1).apply(np.prod) - 1
        # # ticker 2.2 . momentum J=7
        dly_vars.loc[dly_vars.Ticker == ticker, 'J_7'] = (dly_vars.loc[dly_vars.Ticker == ticker, 't-1'] + 1).shift(
            1).rolling(7 - 1).apply(np.prod) - 1
        # # ticker 2.3 . momentum J=15
        dly_vars.loc[dly_vars.Ticker == ticker, 'J_15'] = (dly_vars.loc[dly_vars.Ticker == ticker, 't-1'] + 1).shift(
            1).rolling(15 - 1).apply(np.prod) - 1
        # # ticker 2.4 . momentum J=30
        dly_vars.loc[dly_vars.Ticker == ticker, 'J_30'] = (dly_vars.loc[dly_vars.Ticker == ticker, 't-1'] + 1).shift(
            1).rolling(30 - 1).apply(np.prod) - 1
        # ticker 3 . volatility t-7:t-1
        dly_vars.loc[dly_vars.Ticker == ticker, 'J_std_3'] = (dly_vars.loc[dly_vars.Ticker == ticker, 't-1'] + 1).shift(
            1).rolling(3 - 1).std()
        dly_vars.loc[dly_vars.Ticker == ticker, 'J_std_7'] = (dly_vars.loc[dly_vars.Ticker == ticker, 't-1'] + 1).shift(
            1).rolling(7 - 1).std()
        dly_vars.loc[dly_vars.Ticker == ticker, 'J_std_15'] = (dly_vars.loc[dly_vars.Ticker == ticker, 't-1'] + 1).shift(
            1).rolling(15 - 1).std()
        dly_vars.loc[dly_vars.Ticker == ticker, 'J_std_30'] = (dly_vars.loc[dly_vars.Ticker == ticker, 't-1'] + 1).shift(
            1).rolling(30 - 1).std()
        dly_vars.loc[dly_vars.Ticker == ticker, 't-1'] = dly_vars.loc[dly_vars.Ticker == ticker, 't-1'].shift(1)

        
        #mkt vars
        dly_vars.loc[dly_vars.Ticker == ticker, 'mkt_t-1'] = mkt_ret.values
        dly_vars.loc[dly_vars.Ticker == ticker, 'mkt_J_ret_3'] = mkt_cum_3.values
        dly_vars.loc[dly_vars.Ticker == ticker, 'mkt_J_ret_7'] = mkt_cum_7.values
        dly_vars.loc[dly_vars.Ticker == ticker, 'mkt_J_ret_15'] = mkt_cum_15.values
        dly_vars.loc[dly_vars.Ticker == ticker, 'mkt_J_ret_30'] = mkt_cum_30.values
        dly_vars.loc[dly_vars.Ticker == ticker, 'mkt_J_std_3'] = mkt_std_3.values
        dly_vars.loc[dly_vars.Ticker == ticker, 'mkt_J_std_7'] = mkt_std_7.values
        dly_vars.loc[dly_vars.Ticker == ticker, 'mkt_J_std_15'] = mkt_std_15.values
        dly_vars.loc[dly_vars.Ticker == ticker, 'mkt_J_std_30'] = mkt_std_30.values
         # target returns
    for ticker in tickers:
        # target
        dly_vars.loc[dly_vars.Ticker == ticker, 'target'] = \
            dly_vars.loc[dly_vars.Ticker == ticker, 'Price'].shift(-K) / dly_vars.loc[dly_vars.Ticker == ticker, 'Price'] - 1

    # drop nan
    dly_vars = dly_vars.dropna(axis = 'index')
    dly_target = dly_vars.target
    dly_target = np.ravel(dly_target)

    # returns over past J days
    dly_vars = dly_vars.sort_values(by='Date').reset_index(drop=True)
    dly_data = dly_vars
    dly_vars = dly_vars.drop(['Ticker', 'Date'], axis = 1)
    dly_data = dly_data.set_index(dly_data.Date)

    # MARKET FEATURES
    dly_data = dly_data.drop(['mkt_t-1', 'mkt_J_ret_3', 'mkt_J_ret_7', 'mkt_J_ret_15', 'mkt_J_ret_30', 'mkt_J_std_3',
                              'mkt_J_std_7', 'mkt_J_std_15', 'mkt_J_std_30'], axis=1)
    
    return dly_vars, dly_data, daily_pct_change, daily_close_px, dly_target

def train_test_split(dly_data, daily_close_px):
    start_new = dly_data.Date.index[0]
    end_new = dly_data.index[-1]
    dates = daily_close_px.reset_index()
    dates = dates.set_index(dates.Date)
    dates = dates.Date.loc[start_new:end_new]

    # Chronological split
    test_size = int(len(dates) * 0.2)
    dates_train = dates[:-test_size]
    dates_test = dates[-test_size:]

    X_train = dly_data.loc[dates_train.values, :]
    X_train_old = X_train
    data_train = X_train
    X_train = X_train.reset_index(drop=True)
    X_train = X_train.drop(['Ticker', 'Date'], axis=1)
    y_train = X_train['target']
    X_train = X_train.drop(['target'], axis=1)

    X_test = dly_data.loc[dates_test.values, :]
    data_test = X_test
    X_test = X_test.reset_index(drop=True)
    X_test = X_test.drop(['Ticker', 'Date'], axis=1)
    y_test = X_test.target
    X_test = X_test.drop(['target'], axis=1)

    # Drop Price
    X_train = X_train.drop(['Price'], axis=1)
    X_test = X_test.drop(['Price'], axis=1)

    return X_train, y_train, X_test, y_test, data_train, data_test

def Momentum(pred_r, date, K):
    ret = pred_r.loc[date].reset_index()
    ret['quantile'] = pd.qcut(ret.iloc[:,1].rank(method='first'), 3, labels=False)

    winners = ret[ret['quantile'] == 2]
    losers = ret[ret['quantile'] == 0]

    winnerret = daily_close_px.loc[date + relativedelta(days=K), daily_close_px.columns.isin(winners.Ticker)] / daily_close_px.loc[date, daily_close_px.columns.isin(winners.Ticker)] - 1
    loserret = daily_close_px.loc[date + relativedelta(days=K), daily_close_px.columns.isin(losers.Ticker)] / daily_close_px.loc[date, daily_close_px.columns.isin(losers.Ticker)] - 1

    Momentumprofit = winnerret.mean() - loserret.mean()

    return Momentumprofit

def MOM_Profit(returns, K):
    profits = []
    dates = []
    # run profit
    for date in returns.index[:-K]:
        profits.append(Momentum(returns, date, K))
        dates.append(date)

    frame = pd.DataFrame({'MomentumProfit': profits}, index=dates)
    cum_frame = frame.cumsum() * 100
    profit = frame.MomentumProfit.sum() * 100

    return frame, cum_frame, profit

def plot_feature_importances_cancer(model, name):
    importance = model.feature_importances_
    columns = X_train.columns
    Graph = pd.Series(importance, columns)
    #Graph.sort_values()
    Graph.plot.barh(color='red')
    plt.title('Feature importances for' + name)
    plt.show()

def PURE_MOM():
    MOM_3 = dly_data.pivot(index='Date', columns='Ticker', values='J_3')
    MOM_3_frame, MOM_3_cum_frame, MOM_3_profit = MOM_Profit(MOM_3, 7)
    print('3/7 profit: ', MOM_3_profit, 'weekly r%: ', MOM_3_frame.MomentumProfit.mean() * 7, 'SD: ', MOM_3_frame.MomentumProfit.std())

    MOM_7 = dly_data.pivot(index='Date', columns='Ticker', values='J_7')
    MOM_7_frame, MOM_7_cum_frame, MOM_7_profit = MOM_Profit(MOM_7, 7)
    print('7/7 profit: ', MOM_7_profit, 'weekly r%: ', MOM_7_frame.MomentumProfit.mean() * 7, 'SD: ', MOM_7_frame.MomentumProfit.std())

    MOM_15 = dly_data.pivot(index='Date', columns='Ticker', values='J_15')
    MOM_15_frame, MOM_15_cum_frame, MOM_15_profit = MOM_Profit(MOM_15, 7)
    print('15/7 profit: ', MOM_15_profit, 'weekly r%: ', MOM_15_frame.MomentumProfit.mean() * 7, 'SD: ', MOM_15_frame.MomentumProfit.std())

    MOM_30 = dly_data.pivot(index='Date', columns='Ticker', values='J_30')
    MOM_30_frame, MOM_30_cum_frame, MOM_30_profit = MOM_Profit(MOM_30, 7)
    print('30/7 profit: ', MOM_30_profit, 'weekly r%: ', MOM_30_frame.MomentumProfit.mean() * 7, 'SD: ', MOM_30_frame.MomentumProfit.std())

    
# Prepare variables
dly_vars, dly_data, daily_pct_change, daily_close_px, dly_target = prep_data(tickers, start, end, K=7)
X_train, y_train, X_test, y_test, data_train, data_test = train_test_split(dly_data, daily_close_px)

# Scale datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler_MinMax = MinMaxScaler()
scaler_Standard = StandardScaler()

scaler_MinMax.fit(X_train)
X_train_scaled_MinMax = pd.DataFrame(data=scaler_MinMax.transform(X_train), columns=X_train.columns)
X_test_scaled_MinMax = pd.DataFrame(data=scaler_MinMax.transform(X_test), columns=X_test.columns)

scaler_Standard.fit(X_train)
X_train_scaled_Standard = pd.DataFrame(data=scaler_Standard.transform(X_train), columns=X_train.columns)
X_test_scaled_Standard = pd.DataFrame(data=scaler_Standard.transform(X_test), columns=X_test.columns)

X_train_nonscaled = X_train
X_test_nonscaled = X_test

X_train = X_train_scaled_Standard
X_test = X_test_scaled_Standard

from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train_nonscaled, y_train)
print(f"Accuracy on training set (OLS): {(lr.score(X_train_nonscaled, y_train) * 100):.2f} %")
print(f"Accuracy on test set (OLS): {(lr.score(X_test_nonscaled, y_test) * 100):.2f} %")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
forest = RandomForestRegressor(n_estimators=450, max_features=3, max_depth=4, min_samples_split=3, min_samples_leaf =3, n_jobs=-1) #optimized params
forest.fit(X_train_nonscaled, y_train)
print(f"Accuracy on training set (rf): {(forest.score(X_train_nonscaled, y_train) * 100):.2f} %")
print(f"Accuracy on test set (rf): {(forest.score(X_test_nonscaled, y_test) * 100):.2f} %")
plot_feature_importances_cancer(forest, ' Random Forest')

#Paramerter optimization
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train_nonscaled, y_train)
print(f"Accuracy on training set (rf_random): {(rf_random.score(X_train_nonscaled, y_train) * 100):.3f} %")
print(f"Accuracy on test set (rf_random): {(rf_random.score(X_test_nonscaled, y_test) * 100):.3f} %")

plot_feature_importances_cancer(forest, ' Random Forest')

param_grid = {
    'n_estimators': [300],
    'max_features': [2, 3, 6],
    'max_depth': [5, 15, 30],
    'min_samples_split': [5, 15, 30],
    'min_samples_leaf': [5, 15, 30],
    'max_leaf_nodes' : [5, 15, 30],
    'n_jobs' : [-1]
}

grid = GridSearchCV(rf, param_grid, n_jobs= -1, cv=5)
grid.fit(X_train_nonscaled, y_train)
print(grid.best_params_)

print(f"Accuracy on training set (rf_grid): {(grid.score(X_train_nonscaled, y_train) * 100):.3f} %")
print(f"Accuracy on test set (rf_grid): {(grid.score(X_test_nonscaled, y_test) * 100):.3f} %")