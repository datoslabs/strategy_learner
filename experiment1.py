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
GT User ID: omitted
GT ID: omitted (replace with your GT ID)
"""

import matplotlib.pyplot as plt
import datetime as dt
import ManualStrategy
import StrategyLearner
import marketsimcode as ms
import random
import numpy as np
import pandas as pd


def author():
    return 'omitted'


if __name__ == '__main__':
    random.seed(230)
    np.random.seed(230)
    startdt = dt.datetime(2008, 1, 1)
    enddt = dt.datetime(2009, 12, 31)
    syms = 'JPM'

    mst = ManualStrategy.ManualStrategy()
    learner = StrategyLearner.StrategyLearner()

    learner.addEvidence(syms, startdt, enddt, sv=100000)
    trading_sl = learner.testPolicy(syms, startdt, enddt, sv=100000)
    orders_sl = ms.shape_df_trades(trading_sl)
    sl_val = ms.compute_portvals(orders_sl, start_val=100000, commission=0.00, impact=0.000)
    cr_sl, sddr_sl, adr_sl = ms.assess_portfolio_mini(sl_val)

    trading_mst = mst.testPolicy(syms, sd=startdt, ed=enddt, sv=100000)
    orders_mst = ms.shape_df_trades(trading_mst)
    mst_val = ms.compute_portvals(orders_mst, start_val=100000, commission=0.00, impact=0.000)
    cr_mst, sddr_mst, adr_mst = ms.assess_portfolio_mini(mst_val)

    mst_norm = ms.get_norm_data(mst_val)
    sl_norm = ms.get_norm_data(sl_val)

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.plot(mst_norm, label='Manual Strategy', color='k')
    ax1.plot(sl_norm, label='Strategy Learner', color='r')
    ax1.set_ylabel('Portfolio Value')
    plt.legend(title='Legend')
    plt.title('Manual Strategy vs. Bagged Random Forest Regression Learner')
    fig.savefig('experiment1')

    performance_table = pd.DataFrame(data=[int(mst_val.iloc[-1]), int(sl_val.iloc[-1])],
                                     index=['Manual Strategy', 'Strategy Learner'],
                                     columns=['Portfolio Value'])
    performance_table['Cumulative Return'] = [cr_mst, cr_sl]
    performance_table['Volatility'] = [sddr_mst, sddr_sl]
    performance_table['Avg. Daily Return'] = [adr_mst, adr_sl]

    performance_table.to_csv('experiment1_tbl.csv')
