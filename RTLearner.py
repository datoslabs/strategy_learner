"""  		   	  			    		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
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
"""
import numpy as np
from scipy import stats


class RTLearner(object):

    # I am initializing a tree in the init function, so that I can test some trees.
    def __init__(self, verbose=False, leaf_size=19):
        self.rttree = None  # We're creating a placeholder for tree
        self.leaf_size = leaf_size  # Defining the leaf_size
        self.verbose = verbose

    # Adding my Georgia Tech ID
    def author(self):
        return 'pbaginski3'

    # This is the function to build the data set which we will pass to the build_tree function
    def addEvidence(self, dataX, dataY):
        data = np.concatenate((dataX, np.reshape(dataY, (dataX.shape[0], 1))),
                              axis=1)  # I am using a concatenated dataset instead of X and Y
        self.rttree = self.build_tree(data)  # Calling the function from the Professors slide's to build the tree

    # This is my function to determine the best feature to split on.
    def rand_feature(self, data):
        X = data[:, :-1]  # Extracting features from the data set
        return int(np.random.choice(X.shape[1]))

    # This is the build tree function exactly as described in the professors slides, however, using np.mean instead of
    # median due to regression
    def build_tree(self, data):
        # Copying algorithm from the professors slides, replacing the categorical values with mean values for the leafs
        if data.shape[0] <= self.leaf_size:  # Not sure what this one does
            return np.array([['leaf', np.mean(data[:, -1]), np.nan, np.nan]])  # Adding a leaf node to the tree array
        if np.all(data[:, -1] == data[0, -1]):  # If all values the same, then add another leaf node
            return np.array([['leaf', np.mean(data[:, -1]), np.nan, np.nan]])  # Adding a leaf node to the tree array
        else:
            if np.median(data[:, int(self.rand_feature(data))]) == np.sort(
                    data[:, int(self.rand_feature(data))])[::-1][0]:
                return np.array([['leaf', np.mean(data[:, -1]), np.nan, np.nan]])
                # Return a leaf when maximum recursion occurs, Piazza post @714
            fork = int(self.rand_feature(data))  # Which random feature to split on?
            splitval = np.median(data[:, fork])
            lefttree = self.build_tree(data[data[:, fork] <= splitval])  # Building the left tree
            righttree = self.build_tree(data[data[:, fork] > splitval])  # Building the right tree
            root = np.array([[fork, splitval, 1, lefttree.shape[0] + 1]])  # Building the root
            lft_tree_append = np.append(root, lefttree, axis=0)  # Appending the left tree to the root
            full_tree = np.append(lft_tree_append, righttree, axis=0)  # Appending the right tree to the array
            return full_tree

    # Function to construct a list of predicted values depending on the input
    def query(self, points):
        def predictions(tree_point):
            pred_point = 0
            while not self.rttree[pred_point, 0] == 'leaf':  # Checking wether we have a leaf or not
                fork = self.rttree[pred_point, 0]  # Looking up the value at the fork
                splitval = self.rttree[pred_point, 1]  # Looking up the split value
                if tree_point[int(float(fork))] <= float(splitval):  # Checking which leaf to look at, right or left
                    pred_point = pred_point + int(
                        float(self.rttree[pred_point, 2]))  # Changing the pred row where to find the Y value
                elif tree_point[int(float(fork))] > float(splitval):
                    pred_point = pred_point + int(
                        float(self.rttree[pred_point, 3]))  # Changing the pred row where to find the Y value
            return self.rttree[pred_point, 1]  # Return the prediction value Y

        preds = []  # Creating an empty list where we will attach the predicted values to
        for p in range(points.shape[0]):  # Iterating over the list of values that needs to be predicted
            Y_pred = float(predictions(points[p, :]))  # Retrieving the predictions
            preds.append(Y_pred)  # Appending the retrieved value
        return preds  # Spit back out the list of predictions
