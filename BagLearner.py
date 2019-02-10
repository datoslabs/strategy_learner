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


class BagLearner(object):

    # I am initializing a tree in the init function, so that I can test some trees.
    def __init__(self, learner, kwargs, bags, boost, verbose):
        # Copying the professors code from the assignment
        self.learner = learner
        self.verbose = verbose
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.learners = []
        for i in range(0, self.bags):
            self.learners.append(self.learner(**kwargs))

    # Adding my Georgia Tech ID
    def author(self):
        return 'pbaginski3'

    # As found in chosen answer under for np random choice
    # https://stackoverflow.com/questions/14262654/numpy-get-random-set-of-rows-from-2d-array
    def addEvidence(self, dataX, dataY):
        model_storage = np.zeros((len(self.learners), 3))
        for learner in self.learners:
            X = dataX[np.random.choice(dataX.shape[0], dataX.shape[0], replace=True), :]
            Y = dataY[np.random.choice(dataY.shape[0], dataY.shape[0], replace=True)]
            learner.addEvidence(X, Y)
            np.append(model_storage, learner)

    # Function to construct a list of predicted values depending on the input
    def query(self, points):
        baglearners_out = []
        for i in self.learners:
            individual_out = i.query(points)
            baglearners_out.append(individual_out)
        final = np.mean(baglearners_out, axis=0)
        return final
