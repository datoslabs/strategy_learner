"""MC2-P1: Market simulator.

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Patrick Baginski (replace with your name)
GT User ID: pbaginski3
GT ID: 903383289 (replace with your GT ID)
"""

import pandas as pd
from util import get_data


def author():
    return 'pbaginski3'


def get_norm_data(data):
    return data/data.iloc[0, :]


# need to build function for computing portfolio statistics
def assess_portfolio_mini(strategy_output):
    returns = strategy_output.pct_change()
    returns.iloc[0, :] = 0
    dly_rets = returns[1:]
    cr = (float(strategy_output.iloc[-1, 0]) / float(strategy_output.iloc[0, 0])) - 1.0
    adr = dly_rets.iloc[:, 0].mean()
    sddr = dly_rets.iloc[:, 0].std()
    return cr, sddr, adr


def get_daily_returns(df):
    returns = df.pct_change()
    returns.iloc[0, :] = 0
    returns = returns[1:]
    return returns


def hold_benchmark(symbol, start_date, end_date, stocks=1000):
    df = get_data([symbol], pd.date_range(start_date, end_date), addSPY=False)
    df = df.dropna()
    cols = ['Date', symbol]
    benchmark = pd.DataFrame(data=[(df.index.min(), stocks),
                                (df.index.max(), int(0))], columns=cols)
    benchmark.set_index("Date", inplace=True)
    return benchmark


def shape_df_trades(df):
    column = df.columns.values
    df['Date'] = df.index
    df['Symbol'] = column[0]
    df['Order'] = 'HOLD'
    df['Shares'] = df[column]
    df.reset_index(drop=True, inplace=True)

    df_v2 = df.copy()
    for i in range(0, len(df_v2)):
        if float(df_v2.iloc[i, 0]) < 0:
            df_v2.iloc[i, 3] = 'SELL'
            df_v2.iloc[i, 4] = float(df_v2.iloc[i, 4]) * -1
        elif float(df_v2.iloc[i, 0]) > 0:
            df_v2.iloc[i, 3] = 'BUY'

    cols = df_v2.columns[0]
    df_v2.drop(cols, axis=1, inplace=True)
    return df_v2


def compute_portvals(orders_file, start_val=1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input

    # Reading in the data frame of orders and creating prices as per piazza post @1593
    all_orders = pd.DataFrame(orders_file)
    all_orders.set_index('Date', drop=True, inplace=True)
    # Reading the order
    all_orders.sort_index(ascending=True, inplace=True)  # Sorting the data frame
    start_date = all_orders.index.min()  # determining the start date as per assignment description
    end_date = all_orders.index.max()  # determining end date - assignment says to chose
    # start/end based on order file

    # Building the stock prices data frame - we need to read only trading days based on
    # whats available in SPY
    stocks = all_orders.Symbol.unique()  # getting all the stocks that are in the orders
    # file without duplicates
    stocks_list = stocks.tolist()
    # adding them to a list so we can use the util function
    prices = get_data(stocks_list, pd.date_range(start_date, end_date))  # getting all the stock prices
    prices.dropna(subset=['SPY'])  # Dropping all days where SPY wasn't traded, speak: non-trading days
    prices = prices.drop(['SPY'], axis=1)  # Dropping the SPY column
    prices = prices.fillna(method='ffill')  # Filling forward as per assignment instructions
    prices_clean = prices.fillna(method='bfill')  # Filling backward and storing the final prices
    # dataframe as clean
    prices_clean['cash'] = float(1)  # Adding the cash column as per the youtube video
    stocks_list_v2 = list(stocks_list)  # I initially created this one because I used SPX
    # instead of SPY but was wrong

    # Creating a new data frame from the same structure as the prices data frame
    trades = prices_clean.copy()  # copying the data frame with the same structure as per the
    # piazza post @1593
    trades[prices_clean.columns] = float(0)  # filling all columns with 0s

    for dt, ords in all_orders.iterrows():  # starting to iterate over the rows, in order to get
        # the date and stock price
        if ords['Order'] == 'BUY':  # Determining what happens when BUY, since there are only BUY SELL
            stock = ords['Symbol']  # determining which stock price to pull
            trades.loc[dt, stock] = trades.loc[dt, stock] + ords['Shares']
            # This is to calculate the new number of stocks
            trades.loc[dt, 'cash'] = trades.loc[dt, 'cash'] - prices_clean.loc[dt, stock] * (
                    ords['Shares']) - (commission + impact * prices_clean.loc[dt, stock] *
                                       ords['Shares'])
            # Cost of trades
        else:  # here we determine the sell conditions
            stock = ords['Symbol']  # same procedure as above, selecting the stock price to pick
            trades.loc[dt, stock] = trades.loc[dt, stock] - ords['Shares']  # determining the new
            # volume of stocks
            trades.loc[dt, 'cash'] = trades.loc[dt, 'cash'] + prices_clean.loc[dt, stock] * \
                                     ords['Shares'] - (commission + impact *
                                                       prices_clean.loc[dt, stock] * ords['Shares'])
            # cost of trades

    # Building the data frame holdings as per the piazza post @1593
    holdings = prices_clean.copy()  # again, creating a copy as per the piazza post
    holdings[prices_clean.columns] = float(0)  # filling that new copy with 0s
    holdings['cash'].iloc[0] = start_val + trades['cash'].iloc[0]  # as per the piazza post we
    # initiate the cash as start val
    for stk in stocks_list_v2:  # for steaks in stocks ... basically picking the stocks that are
        # not SPY
        holdings[stk].iloc[0] = holdings[stk].iloc[0] + trades[stk].iloc[0]  # and summing it up
        # with trades as per youtube

    for r in range(1, len(holdings.index)):  # now we are looing at the cash value and do
        # the operation as in youtube
        holdings['cash'].iloc[r] = holdings['cash'].iloc[r - 1] + trades['cash'].iloc[r]  # new cash
        # value
        for stk in stocks_list_v2:  # dont select the SPY columna
            holdings[stk][r] = holdings[stk].iloc[r - 1] + trades[stk].iloc[r]  # new stock volume

    # Building the data frame values as per the Youtube video / @1593 picture of in-person lecture
    values = prices_clean * holdings  # this one is simple and basically copied from the
    # youtube presentation

    # Building the final portfolio value as per
    # https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-then-filling-it
    portvals = pd.DataFrame(index=values.index, columns=['portvals'])  # how do I
    # fill a dataframe? stack overflow post
    for r in range(len(portvals)):  # getting the sum of all shares in the portfolio
        portvals['portvals'].iloc[r] = values.iloc[r].sum()  # more sums for more shares

    return portvals
