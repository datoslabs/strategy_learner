"""  		   	  			    		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
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
  		   	  			    		  		  		    	 		 		   		 		  
Student Name: omitted (replace with your name)
"""

import datetime as dt
import pandas as pd
import BagLearner as BG
import RTLearner as RT
from indicators import bollinger_value, rate_of_change, relative_strength_index
from util import get_data


class StrategyLearner(object):

    def __init__(self, verbose=False, impact=0.000):
        self.verbose = verbose
        self.impact = impact

    def author(self):
        return 'omitted'

    def create_learning_dataframe(self, stock_prices):
        symbol = stock_prices.columns[1]
        indicator_span = 20
        roc_span = 20
        n = 20

        days_prior = int(60)
        sd = stock_prices.index[0]
        ed = stock_prices.index[-1]
        sd_adj = sd - dt.timedelta(days=days_prior)
        prices = get_data([symbol], pd.date_range(sd_adj, ed))
        prices.dropna(subset=['SPY'])
        prices = prices.drop(['SPY'], axis=1)
        prices = prices.fillna(method='ffill')
        prices_clean = prices.fillna(method='bfill')

        b_val_sl = bollinger_value(prices_clean, window=indicator_span)
        roc = rate_of_change(prices_clean, size=roc_span)
        rsi = relative_strength_index(prices_clean, period=indicator_span)
        new_df = b_val_sl.join(rsi, how='inner')
        new_df = new_df.join(roc, how='inner')

        ret = (prices_clean.shift(-n)/prices_clean) - 1.0
        ret = ret.dropna()
        y_df = pd.DataFrame(data=ret, index=ret.index)
        y_df = y_df.loc[sd:]

        train_df = new_df.join(y_df, how='inner')

        return train_df

    def addEvidence(self, symbol="IBM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), sv=10000):

        features = ['roc', 'rsi', 'bval']
        bags = 20
        dates = pd.date_range(sd, ed)
        data = get_data([symbol], dates)
        train_df = self.create_learning_dataframe(data)
        train_x = train_df[features].as_matrix()
        train_y = train_df.iloc[:, -1:].as_matrix()

        self.bg_learner = BG.BagLearner(learner=RT.RTLearner, kwargs={}, bags=bags, boost=False, verbose=False)
        self.bg_learner.addEvidence(train_x, train_y)

    def testPolicy(self, symbol="IBM", sd=dt.datetime(2009, 1, 1), ed=dt.datetime(2010, 1, 1), sv=10000):

        YBUY = 0.003
        YSELL = -1 * YBUY
        dates = pd.date_range(sd, ed)
        data = get_data([symbol], dates)
        test_df = self.create_learning_dataframe(data)
        test_x = test_df.iloc[:, :-1].as_matrix()

        preds_y = self.bg_learner.query(test_x)

        trades = test_df.iloc[:, -1:].copy()
        for i in range(len(trades)):
            if (preds_y[i] - self.impact) > YBUY:
                trades.iloc[i, 0] = int(1000)
                continue
            if (preds_y[i] - self.impact) < YSELL:
                trades.iloc[i, 0] = -int(1000)
                continue
            else:
                trades.iloc[i, 0] = 0
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
