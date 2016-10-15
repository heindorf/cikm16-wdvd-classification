# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2016 Stefan Heindorf, Martin Potthast, Benno Stein, Gregor Engels
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------


import abc
import logging

import numpy as np
import pandas as pd
import sklearn.base

_logger = logging.getLogger()


###################################################
# Used for SIL (single instance learning
###################################################
class BaseMultipleInstanceClassifier(sklearn.base.BaseEstimator):
    __metaclass__ = abc.ABCMeta
       
    @abc.abstractmethod
    def fit(self, g, X, y):
        pass
    
    @abc.abstractmethod
    def predict_proba(self, g, X):
        pass


class SingleInstanceClassifier(BaseMultipleInstanceClassifier):
    def __init__(self, base_estimator, agg_func='mean'):
        self._agg_func = agg_func  # name of aggregation function
        self._base_estimator = base_estimator
        self._proba = None
        
    def fit(self, g, X, y):
        self._base_estimator.fit(X, y)
        self._proba = None
    
    def set_proba(self, proba):
        self._proba = proba
    
    # g contains the group ids
    def predict_proba(self, g, X):
        # Determines the aggregation function (e.g., mean, max, min, ...)
        if len(g) != len(X):
            raise Exception("g and X should have same lenght")
        
        agg_func = self._get_agg_func(self._agg_func)
            
        # Has user explicitly specified proba?
        # (to save some computational time)
        if self._proba is not None:
            proba = self._proba  # use stored proba and ignore X
        else:
            # if proba has not been explicitly set,
            # use base_estimator to compute it
            proba = self._base_estimator.predict_proba(X)
            
        agg_proba = self._agg_proba(agg_func, g, proba)
        
        return agg_proba
    
    @staticmethod
    def _get_agg_func(agg_func):
        
        if agg_func == 'mean':
            result = pd.core.groupby.GroupBy.mean
        elif agg_func == 'min':
            result = pd.core.groupby.GroupBy.min
        elif agg_func == 'max':
            result = pd.core.groupby.GroupBy.max
        elif agg_func == 'median':
            result = pd.core.groupby.GroupBy.median
        elif agg_func == 'first':
            pd.core.groupby.GroupBy.first
        elif agg_func == 'last':
            pd.core.groupby.GroupBy.last
        else:
            raise Exception("Unknown function name: " + str(agg_func))
            
        return result
            
    @staticmethod
    def _agg_proba(function, group, proba):
        tmp = pd.DataFrame()
        tmp['group'] = group
        tmp['proba'] = proba
        
        agg_proba = function(tmp.groupby('group')['proba'])
        agg_proba.name = 'agg_proba'
        tmp = tmp.join(agg_proba, on='group', how='left')
        
        result = tmp['agg_proba'].values
        
        return result
        
    
class SimpleMultipleInstanceClassifier(BaseMultipleInstanceClassifier):
    def __init__(self, base_estimator, trans_func='min_max'):
        self._trans_func = trans_func  # name of aggregation function
        self._base_estimator = base_estimator
        
    def fit(self, g, X, y):
        trans_func = self._get_trans_func(self._trans_func)
        _, trans_X, trans_y = trans_func(g, X, y)
        self._base_estimator.fit(trans_X, trans_y)
        
    def predict_proba(self, g, X):
        trans_func = self._get_trans_func(self._trans_func)
        
        # transformation into 'group space'
        trans_g, trans_X, _ = trans_func(g, X, None)
        
        # prediction in 'group space'
        trans_proba = self._base_estimator.predict_proba(trans_X)
        
        # transformation into 'instance space'
        proba = self._transform_back(trans_g, trans_proba, g)
    
        return proba
    
    @classmethod
    def _get_trans_func(cls, trans_func):
        if trans_func == 'min_max':
            result = cls._min_max_trans_func
        elif trans_func == 'mean':
            result = cls._mean_trans_func
        else:
            raise Exception("Unknown function name: " + str(trans_func))
            
        return result
    
    # Idea taken from Jaume Amores, 2013
    @classmethod
    def _min_max_trans_func(cls, g, X, y=None):
        if y is not None:
            df_y = pd.DataFrame(y)
            # df_y.columns = ['rollbackReverted']
            df_y.insert(0, 'revisionGroup', g)
            result_y = cls._group(df_y, pd.core.groupby.GroupBy.max, '_max')
            result_y = result_y.iloc[:, [1]]
            result_y = np.ascontiguousarray(result_y.values.ravel())
        else:
            result_y = None
            
        df_X = pd.DataFrame(X)
        
        # index of revisionGroup must start at 0
        df_X.insert(0, 'revisionGroup', g)
        max_X = cls._group(df_X, pd.core.groupby.GroupBy.max, '_max')
        min_X = cls._group(df_X, pd.core.groupby.GroupBy.min, '_min')
        result_groupIDs = max_X.iloc[:, [0]]
        result_X = pd.concat([max_X.iloc[:, 1:], min_X.iloc[:, 1:]], axis=1)
                
        result_groupIDs = np.ascontiguousarray(result_groupIDs.values)
        result_X = np.ascontiguousarray(result_X.values)
        
        return result_groupIDs, result_X, result_y
    
    # Assumption: the first column of the data frame is  revisionGroup
    @classmethod
    def _group(cls, df_X, agg_func, suffix):
        result_df = agg_func(df_X.groupby('revisionGroup', as_index=False))
        result_df.columns.values[1:] = cls._concat_str(df_X.columns[1:].values,
                                                       suffix)
        
        return result_df
    
    @staticmethod
    def _concat_str(list2, str2):
        result = []
        
        for item in list2:
            result.append(str(item) + str2)
        return result
    
    @staticmethod
    def _transform_back(trans_g, trans_proba, g):
        trans_proba = pd.DataFrame(trans_proba)
        
        groups = pd.DataFrame(trans_g, columns=['groups'])
        groups = pd.concat([groups, trans_proba], axis=1)
        groups.name = 'groups'
        
        instances = pd.DataFrame(g, columns=['groups'])
        instances.name = 'instances'
                                                   
        df = instances.merge(groups, on='groups', how='left')
        
        result_proba = np.ascontiguousarray(df.iloc[:, 1:].values)
        
        result_proba = result_proba[:, 1]
        
        return result_proba


class CombinedMultipleInstanceClassifier(BaseMultipleInstanceClassifier):
    def __init__(self, base_estimator):
        if base_estimator is not None:
            self._sil_clf = SingleInstanceClassifier(
                base_estimator=sklearn.base.clone(base_estimator))
            self._smi_clf = SimpleMultipleInstanceClassifier(
                base_estimator=sklearn.base.clone(base_estimator))
    
    def fit(self, g, X, y):
        self._sil_clf.fit(g, X, y)
        self._smi_clf.fit(g, X, y)
        
    def set_proba(self, si_proba, smi_proba):
        self._sil_proba = si_proba
        self._smi_proba = smi_proba
        
    def predict_proba(self, g, X):
        if self._sil_proba is None:
            sil_proba = self._sil_clf.predict_proba(g, X)
        else:
            sil_proba = self._sil_proba
            
        if self._smi_proba is None:
            smi_proba = self._smi_clf.predict_proba(g, X)
        else:
            smi_proba = self._smi_proba
        
        proba = self.average_proba(sil_proba, smi_proba)
        return proba
    
    # Averages the scores of two classifiers
    @staticmethod
    def average_proba(prob1, prob2):
        tmp = pd.DataFrame()
        tmp['prob1'] = prob1
        tmp['prob2'] = prob2
        
        avg_proba = np.ascontiguousarray(tmp.mean(axis=1).values)
        
        return avg_proba
