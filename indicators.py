"""
Student Name: Patrick Baginski (replace with your name)
GT User ID: pbaginski3
GT ID: 903383289 (replace with your GT ID)"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import marketsimcode as ms
from util import get_data


def author():
    return 'pbaginski3'


def bollinger_value(df, window=20):
    r_mean = df.rolling(window=window).mean()
    r_std = df.rolling(window=window).std()
    b_val = (df - r_mean) / r_std
    b_val['bval'] = b_val[b_val.columns[0]]
    b_val = b_val.drop([b_val.columns[0]], axis=1)
    return b_val


def relative_strength_index(series, period=14):
    difference = series.diff().dropna()
    df1 = difference * 0
    df2 = df1.copy()
    df1[difference > 0] = difference[difference > 0]
    df2[difference < 0] = -difference[difference < 0]
    df1[df1.index[period-1]] = np.mean(df1[:period])
    df1 = df1.drop(df1.index[:(period-1)])
    df2[df2.index[period-1]] = np.mean(df2[:period])
    df2 = df2.drop(df2.index[:(period-1)])
    rs = df1.ewm(com=period-1, adjust=False).mean() / df2.ewm(com=period-1, adjust=False).mean()
    rsi_v = 100 - 100 / (1 + rs)
    rsi_v['rsi'] = rsi_v.iloc[:, 0]
    rsi_v = rsi_v.drop([rsi_v.columns[0], rsi_v.columns[1]], axis=1)
    return rsi_v


def rate_of_change(stock_series, size=5):
    shifts = stock_series.iloc[:, 0].diff(size)
    adjusted_series = stock_series.iloc[:, 0].shift(size)
    rate_change = pd.Series(shifts/adjusted_series, name='roc')
    rate_change = pd.DataFrame(data=rate_change, index=rate_change.index)
    return rate_change


def plot_roc(stock_price, stock_roc, title='Rate of Change'):
    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.plot(stock_price.index, stock_price, label='Adj Close', color='r')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Adj Close', color='r')
    ax1.tick_params('y', colors='r')

    ax2 = ax1.twinx()
    ax2.plot(stock_roc.index, stock_roc, label='Rate of Change', color='k')
    ax2.set_ylabel('Rate of Change', color='k')
    ax2.tick_params('y', colors='k')

    plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.legend(title='Legend')
    plt.title(title)
    fig.savefig('ROC')


def plot_bvalue(stock_price, b_val, title='Bollinger Value'):
    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.plot(stock_price.index, stock_price, label='Adj Close', color='r')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Adj Close', color='r')
    ax1.tick_params('y', colors='r')

    ax2 = ax1.twinx()
    ax2.plot(b_val.index, b_val, label='Bollinger Value', color='g')
    ax2.set_ylabel('Bollinger Value', color='g')
    ax2.tick_params('y', colors='g')

    plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.legend(title='Legend')
    plt.title(title)
    fig.savefig('Bval')


def plot_rsi(stock_price, stock_roc, title='RSI'):
    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.plot(stock_price.index, stock_price, label='Adj Close', color='r')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Adj Close', color='r')
    ax1.tick_params('y', colors='r')

    ax2 = ax1.twinx()
    ax2.plot(stock_roc.index, stock_roc, label='RSI', color='Blue')
    ax2.set_ylabel('RSI', color='Blue')
    ax2.tick_params('y', colors='Blue')

    plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.legend(title='Legend')
    plt.title(title)
    fig.savefig('RSI')


if __name__ == '__main__':
    startdt = dt.datetime(2008, 1, 1)
    enddt = dt.datetime(2009, 12, 31)
    syms = ['JPM']
    JPM = get_data(syms, pd.date_range(startdt, enddt))
    JPM.dropna(subset=['SPY'])
    JPM = JPM.drop(['SPY'], axis=1)  # Dropping the SPY column
    JPM = JPM.fillna(method='ffill')  # Filling forward as per assignment instructions
    JPM_clean = JPM.fillna(method='bfill')
    JPM_nrm = ms.get_norm_data(JPM_clean)
    b_Val_test = bollinger_value(JPM_nrm)
    roc = rate_of_change(JPM_nrm)
    rsi = relative_strength_index(JPM_nrm)
    plot_bvalue(JPM_nrm, b_Val_test)
    plot_roc(JPM_nrm, roc)
    plot_rsi(JPM_nrm, rsi)
