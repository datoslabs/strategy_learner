# build one more function to plot the indicators (one chart per indicator with benchmrk + stock as comp)

# Which indicators to use? Bollinger bands (BB Squeeze), MACD (Strat ??), SMA (% above Avg SMA Strat),
# EMA, STOCH, CCI (CCI Correct Strat), Momentum (Moving Momentum Strat)

# The strategy I will implement is a combination of %-above-average SMA strategy, BB squeeze and moving ROC
# We start by implementing each of the strategy separate from each other, later we will combine them in some kind of
# hierarchy of combination - based on their individual performance
# Basically it's like: build your indicator based on x days from data, then create order based on indicator, then add
# another day of data to it and repeat, e.g. ROC:
# Calculate ROC on day n + 5, moving up or down? buy or sell!

import pandas as pd
from util import get_data
import datetime as dt
from indicators import rate_of_change, bollinger_value, relative_strength_index


class ManualStrategy(object):
    def __init__(self):
        pass

    def testPolicy(self, symbol, sd, ed, sv):
        days_prior = int(60)
        sd_adj = sd - dt.timedelta(days=days_prior)
        prices = get_data([symbol], pd.date_range(sd_adj, ed))  # getting all the stock prices
        prices.dropna(subset=['SPY'])  # Dropping all days where SPY wasn't traded, speak: non-trading days
        prices = prices.drop(['SPY'], axis=1)  # Dropping the SPY column
        prices = prices.fillna(method='ffill')  # Filling forward as per assignment instructions
        prices_clean = prices.fillna(method='bfill')

        amount = 1000
        trades = prices_clean.loc[sd:].copy()

        # Prices clean now holds multiple days before, we need to make sure we pull from day 1 - whatever is needed
        cnt = 0
        cnt_bval = 0
        cnt_rsi = 0

        for d in trades.index:
            input_rsi = prices_clean.loc[:d]
            rsi = relative_strength_index(input_rsi, period=20)
            rsi_thresh = float(rsi.iloc[-1])
            if trades.iloc[cnt_rsi, 0] < rsi_thresh:
                trades.iloc[cnt_rsi, 0] = float(amount)
                cnt_rsi = cnt_rsi + 1
                continue
            if rsi_thresh < trades.iloc[cnt_rsi, 0]:
                trades.iloc[cnt_rsi, 0] = -float(amount)
                cnt_rsi = cnt_rsi + 1
                continue
            else:
                trades.iloc[cnt, 0] = 0
                cnt_rsi = cnt_rsi + 1
                continue

        for d in trades.index:
            input_roc = prices_clean.loc[:d]
            roc = rate_of_change(input_roc, size=2)
            roc = float(roc.iloc[-1])
            roc_thresh = 0.005
            if roc_thresh <= roc <= -roc_thresh:
                cnt = cnt + 1
                continue
            if roc < roc_thresh:
                trades.iloc[cnt, 0] = float(amount)
                cnt = cnt + 1
                continue
            else:
                trades.iloc[cnt, 0] = -float(amount)
                cnt = cnt + 1
                continue

        for d in trades.index:
            input_bval = prices_clean.loc[:d]
            b_val = bollinger_value(input_bval)
            b_val_a = float(b_val.iloc[-1])
            b_val_thresh = float(2)
            if b_val_thresh <= b_val_a:
                trades.iloc[cnt_bval, 0] = -float(amount)
                cnt_bval = cnt_bval + 1
                continue
            if b_val_a <= -b_val_thresh:
                trades.iloc[cnt_bval, 0] = float(amount)
                cnt_bval = cnt_bval + 1
                continue
            else:
                cnt_bval = cnt_bval + 1
                continue

        df_trades = trades.copy()
        holdings = float(0)

        for i in range(0, len(trades)):
            if holdings == 0:
                holdings = holdings + int(df_trades.iloc[i, 0])
                continue
            if holdings < 0:
                if df_trades.iloc[i, 0] > 0:
                    df_trades.iloc[i, 0] = float(2000)
                    holdings = holdings + float(2000)
                else:
                    df_trades.iloc[i, 0] = float(0)
            else:
                if df_trades.iloc[i, 0] > 0:
                    df_trades.iloc[i, 0] = float(0)
                else:
                    df_trades.iloc[i, 0] = -float(2000)
                    holdings = holdings - float(2000)

        return df_trades
