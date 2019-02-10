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

import matplotlib.pyplot as plt
import datetime as dt
import StrategyLearner
import marketsimcode as ms
import random
import numpy as np
import pandas as pd


def author():
    return 'pbaginski3'


if __name__ == '__main__':
    random.seed(2712)
    np.random.seed(2712)
    startdt = dt.datetime(2008, 1, 1)
    enddt = dt.datetime(2009, 12, 31)
    syms = 'JPM'

    # I would expect it to perform worse because trades won't make sense
    learner_exp1 = StrategyLearner.StrategyLearner()
    learner_exp3 = StrategyLearner.StrategyLearner(impact=0.5)
    learner_exp4 = StrategyLearner.StrategyLearner(impact=0.05)
    learner_exp5 = StrategyLearner.StrategyLearner(impact=0.005)

    learner_exp1.addEvidence(syms, startdt, enddt, sv=100000)
    trading_sl_exp1 = learner_exp1.testPolicy(syms, startdt, enddt, sv=100000)
    orders_sl_exp1 = ms.shape_df_trades(trading_sl_exp1)
    sl_val_exp1 = ms.compute_portvals(orders_sl_exp1, start_val=100000, commission=0.00, impact=0.000)

    learner_exp3.addEvidence(syms, startdt, enddt, sv=100000)
    trading_sl_exp3 = learner_exp3.testPolicy(syms, startdt, enddt, sv=100000)
    orders_sl_exp3 = ms.shape_df_trades(trading_sl_exp3)
    sl_val_exp3 = ms.compute_portvals(orders_sl_exp3, start_val=100000, commission=0.00, impact=0.000)

    learner_exp4.addEvidence(syms, startdt, enddt, sv=100000)
    trading_sl_exp4 = learner_exp4.testPolicy(syms, startdt, enddt, sv=100000)
    orders_sl_exp4 = ms.shape_df_trades(trading_sl_exp4)
    sl_val_exp4 = ms.compute_portvals(orders_sl_exp4, start_val=100000, commission=0.00, impact=0.000)

    learner_exp5.addEvidence(syms, startdt, enddt, sv=100000)
    trading_sl_exp5 = learner_exp5.testPolicy(syms, startdt, enddt, sv=100000)
    orders_sl_exp5 = ms.shape_df_trades(trading_sl_exp5)
    sl_val_exp5 = ms.compute_portvals(orders_sl_exp5, start_val=100000, commission=0.00, impact=0.000)

    sl_norm_exp1 = ms.get_norm_data(sl_val_exp1)
    sl_norm_exp3 = ms.get_norm_data(sl_val_exp3)
    sl_norm_exp4 = ms.get_norm_data(sl_val_exp4)
    sl_norm_exp5 = ms.get_norm_data(sl_val_exp5)

    exp1_cnt = orders_sl_exp1[(orders_sl_exp1.Order != 'HOLD')].count()['Order']
    exp3_cnt = orders_sl_exp3[(orders_sl_exp3.Order != 'HOLD')].count()['Order']
    exp4_cnt = orders_sl_exp4[(orders_sl_exp4.Order != 'HOLD')].count()['Order']
    exp5_cnt = orders_sl_exp5[(orders_sl_exp5.Order != 'HOLD')].count()['Order']

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.plot(sl_norm_exp1, label='0 Impact', color='r')
    ax1.plot(sl_norm_exp3, label='0.5 Impact', color='g')
    ax1.plot(sl_norm_exp4, label='0.05 Impact', color='b')
    ax1.plot(sl_norm_exp5, label='0.005 Impact', color='y')
    ax1.set_ylabel('Portfolio Value')

    plt.legend(title='Legend')
    plt.title('Effect of Impact on Random Forest-Based Portfolio Value')
    fig.savefig('experiment2')

    performance_table = pd.DataFrame(data=[int(sl_val_exp1.iloc[-1]),
                                           int(sl_val_exp3.iloc[-1]),
                                           int(sl_val_exp4.iloc[-1]),
                                           int(sl_val_exp5.iloc[-1])],
                                     index=['Imp. 0.0', 'Imp. 0.5', 'Imp. 0.05', 'Imp. 0.005'],
                                     columns=['Portfolio Value'])
    performance_table['# of trades'] = [exp1_cnt, exp3_cnt, exp4_cnt, exp5_cnt]

    performance_table.to_csv('experiment2_tbl.csv')
