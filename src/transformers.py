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

import logging
import itertools

import numpy as np
import pandas as pd

import scipy.sparse

from numpy.core import getlimits

import sklearn.feature_extraction
import sklearn.grid_search
import sklearn.linear_model
import sklearn.metrics

from sklearn.preprocessing.label import LabelBinarizer
from sklearn.base import TransformerMixin

import config

from .utils import collect_garbage

from . import evaluation
logger = logging.getLogger()

####################################################################
# Transformers
#    - Input: Pandas DataFrame
#    - Output: Either Pandas DataFrame (or scipy.sparse.csr_matrix)

####################################################################
# Handling of NAs
####################################################################


class StringImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.fillna("missing")

 
class ZeroImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        result = X.fillna(0)
        result = InfinityImputer().fit_transform(result)
        
        return result

 
class MinusOneImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        result = X.fillna(-1)
        result = InfinityImputer().fit_transform(result)

        return result

    
class MeanImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.__mean = X.mean()
        return self
    
    def transform(self, X):
        result = X.fillna(self.__mean)
        result = InfinityImputer().fit_transform(result)

        return result

    
class MedianImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.__median = X.median()
        return self
    
    def transform(self, X):
        result = X.fillna(self.__median)
        result = InfinityImputer().fit_transform(result)

        return result


class BooleanImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, **fit_params):
        collect_garbage()
        result = X.astype(np.float32)
        result = result.fillna(0.5)
        
        return pd.DataFrame(result)

   
class InfinityImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        result = X
        
        for column in X.columns:
            datatype = result.loc[:, column].dtype.type
            limits = getlimits.finfo(datatype)
                     
            result.loc[:, column].replace(np.inf, limits.max, inplace=True)
            result.loc[:, column].replace(-np.inf, limits.min, inplace=True)
        
        return result


class NaIndicator(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        result = X.isnull()
        result = result.to_frame()
        
        return result

   
class Float32Transformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        result = X.astype(np.float32)
        
        return result
    
#######################################################
# Scaling
#######################################################


# computes the formula sign(X)*ceil(log2(|X|+1))
class LogTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        result = X

        sign = result.apply(np.sign)
        
        result = result.apply(np.absolute)
        result = result + 1
        result = result.apply(np.log2)
        result = result.apply(np.ceil)
        result = sign * result
        
        result = result.fillna(0)
        return result

  
class SqrtTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        result = X
        
        result = result + 1
        result = result.apply(np.sqrt)
        result = result.apply(np.ceil)
        return result
    
   
#######################################################
# Handling of Strings
#######################################################

class LengthTransformer(TransformerMixin):
    """Computes the length of a string (both in bytes as well as in words)"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if len(X.columns) > 1:
            raise Exception("Only one column supported")
        else:
            X = X.iloc[:, 0]
        
        collect_garbage()
        X = X.astype(str)  # sometimes fails with MemoryError
        
        result = pd.DataFrame()
        result['byteLength'] = X.str.len()
        result['byteLength'].fillna(-1, inplace=True)
        
        rows = X.str.findall("\w+")

        result['wordLength'] = [(len(row) if row == row else -1) for row in rows]
        
        return result
    
########################################################
# Combining features
########################################################


# see https://github.com/pydata/pandas/issues/11635
def category_workaround(X):
    result = X.copy()
    for column in X.columns:
        if hasattr(X[column], 'cat'):
            result[column] = result[column].cat.codes
    return result


class AggTransformer(TransformerMixin):
    """
    Given a data frame with columns (c1, c2, ..., cn), computes for each tuple
    (c1, c2, ..., cn-1), the aggregation of the values in the last column cn.
    
    For example, computes for each user, the average number of bytes added per
    revision (aggregation function is mean).
    
    For example, computers for each user, the unique number of items
    edited (aggregation function is nunique).
    
    """
    def __init__(self, func):
        self.__func = func
        
    def fit(self, X, y=None):
        X = category_workaround(X)
        
        firstColumns = list(X.columns[0:-1])
        lastColumn = X.columns[-1]
        
        grouped = X.groupby(by=firstColumns)[lastColumn]
        
        self.__aggValues = self.__func(grouped)
        self.__aggValues.name = '_aggValues'
        
        return self
    
    def transform(self, X):
        X = category_workaround(X)
        
        firstColumns = list(X.columns[0:-1])
        
        # join first columns with the number of aggregation values in the last
        # column
        result = X.join(self.__aggValues, on=firstColumns, how='left')
        
        # about 10% of users are NA (because they are not in the training set)
        result = result['_aggValues'].fillna(0)
        result = result.to_frame()
        
        return result


class UniqueTransformer(TransformerMixin):
    """
    Given a data frame with columns (c1, c2, ..., cn), computes for each tuple
    (c1, c2, ..., cn-1) the number of unique values in the last column cn.
    
    For example, computes for each user the number of uniques items edited.
    
    """
    def fit(self, X, y=None):
        aggTransformer = AggTransformer(pd.core.groupby.SeriesGroupBy.nunique)  # @UndefinedVariable
        aggTransformer.fit(X, y)
        return aggTransformer

   
class SumTransformer(TransformerMixin):
    def fit(self, X, y=None):
        aggTransformer = AggTransformer(pd.core.groupby.DataFrameGroupBy.sum)  # @UndefinedVariable
        aggTransformer.fit(X, y)
        return aggTransformer

  
class MeanTransformer(TransformerMixin):
    def fit(self, X, y=None):
        aggTransformer = AggTransformer(pd.core.groupby.DataFrameGroupBy.mean)  # @UndefinedVariable
        aggTransformer.fit(X, y)
        return aggTransformer

         
#################################################################
# Handling of categorical values
#################################################################
class FrequencyTransformer(TransformerMixin):
    """
    Given a data frame with columns (C1, C2, ..., Cn), computes for each
    unique tuple (c1, c2, ..., cn), how often it appears in the data frame.
    
    For example, for every revision, it counts how many revisions were done by
    this user (one column C1='userName').
    """
    
    def fit(self, X, y=None):
        self.__frequencies = X.groupby(by=list(X.columns)).size()
        self.__frequencies.name = 'frequencies'
        return self
    
    def transform(self, X):
        result = X.join(self.__frequencies, on=list(X.columns), how='left')
        
        # all other frequencies are at least 1
        result = result['frequencies'].fillna(0)
        result = result.to_frame()

        return result

   
class LongTailImputer(TransformerMixin):
    def __init__(self, count=None):
        self.__count = count
    
    def fit_transform(self, X, y=None, **fit_params):
        topValues = X.groupby(by=X).size()[0:self.__count]
    
        topValues.name = 'topValues'
        
        result = pd.DataFrame(X).join(topValues, on=X.name, how='left')
        result = result['topValues'].fillna(0)
        
        return result

   
class CategoryBinarizer(TransformerMixin):
    def __init__(self):
        self.__encoder = LabelBinarizer(sparse_output=False)
    
    def fit(self, X, y=None):
        # X = X.astype(str)
        X = X.values
        self.__encoder.fit(X)
        return self
        
    def transform(self, X):
        X = X.values
        result = self.__encoder.transform(X)
        result = pd.DataFrame(result)
        result.columns = self.__encoder.classes_
               
        return result


#############################################################
# Logical Transformers
#############################################################
class LogicalOrTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        result = pd.Series([False] * len(X))
        
        for column in X.columns:
            result = result | X[column]
            
        result = result.to_frame()
        
        return result


##############################################################
# Cumulative Transformers (only considering information up
# until the current revision)
##############################################################
class CumFrequencyTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
       
    def transform(self, X):
        # assumption: X is ordered by revisionId
        grouped = X.groupby(by=list(X.columns))
        result = grouped.cumcount() + 1
        
        result = result.to_frame()
        
        return result

   
class CumAggTransformer(TransformerMixin):
    
    # func should be a cumulative function
    def __init__(self, func):
        self.__func = func
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        firstColumns = list(X.columns[0:-1])
        lastColumn = X.columns[-1]
        
        grouped = X.groupby(by=firstColumns)[lastColumn]
        
        # Some functions pandas.core.groupby.DataFrameGroupBy.X are
        # realized as properties
        if isinstance(self.__func, property):
            result = self.__func.fget(grouped)()
        else:
            result = self.__func(grouped)
            
        result = result.fillna(0)
            
        if not isinstance(result, pd.DataFrame):
            result = result.to_frame()
                   
        return result

   
class CumCountTransformer(TransformerMixin):
    def fit(self, X, y=None):
        transformer = CumAggTransformer(pd.core.groupby.DataFrameGroupBy.cumcount)  # @UndefinedVariable
        transformer.fit(X, y)
        return transformer


# not used, because too slow (alternative implementation is used)
class CumSumTransformerPandas(TransformerMixin):
    def fit(self, X, y=None):
        transformer = CumAggTransformer(pd.core.groupby.DataFrameGroupBy.cumsum)  # @UndefinedVariable
        transformer.fit(X, y)
        return transformer

   
class CumMinTransformer(TransformerMixin):
    def fit(self, X, y=None):
        transformer = CumAggTransformer(pd.core.groupby.DataFrameGroupBy.cummin)  # @UndefinedVariable
        transformer.fit(X, y)
        return transformer

   
class CumMaxTransformer(TransformerMixin):
    def fit(self, X, y=None):
        transformer = CumAggTransformer(pd.core.groupby.DataFrameGroupBy.cummax)  # @UndefinedVariable
        transformer.fit(X, y)
        return transformer

   
class CumMeanTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        cumSumTransformer = CumSumTransformer()
        cumCountTransformer = CumCountTransformer()
        
        sums = cumSumTransformer.fit_transform(X)
        counts = cumCountTransformer.fit_transform(X)
        
        result = sums.iloc[:, 0] / (counts.iloc[:, 0] + 1)
        result = result.astype(np.float32)
        result = result.fillna(0)
        result = result.to_frame()
        
        return result


# This class is implemented in pure Python, because the Pandas implementation
# cumsum is too slow (in particular for many groups such as items)
class CumSumTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    # Assumption: X is ordered by revisionId
    # Assumption: X is indexed 0,1,2, ... , len(df)
    def transform(self, X):
        if X.index[len(X) - 1] != len(X) - 1:
            raise Exception("Expecting data frame that is indexed 0,1,2,3, ...")
        
        result = [np.nan] * len(X)
        
        # dictionary of numbers. The dicitonary has the first columns of the
        # dataframe as key and the current sum for those as value.
        dictionary = {}
        
        for row in X.itertuples(index=True):
            index = row[0]
            firstColumns = row[1:-1]
            lastColumn = row[-1]
            
            # Skip firstColumns containing NaN (dictionaries do not work well
            # with np.float(np.NaN) values)
            # Pandas also ignores groups containing NaN
            if any([x != x for x in firstColumns]):
                continue
            
            if not firstColumns in dictionary:
                dictionary[firstColumns] = 0
            
            # Ignore NaN values in last column
            if lastColumn == lastColumn:
                dictionary[firstColumns] += lastColumn
           
            result[index] = dictionary[firstColumns]
        
        result = pd.DataFrame(result)
        
        result = result.fillna(0)  # necessary for firstColumns containing NaN
          
        return result

    
class CumUniqueTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    # Assumption: df is ordered by revisionId
    # Assumption: df is indexed 0,1,2, ... , len(df)
    def transform(self, X):
        if X.index[len(X) - 1] != len(X) - 1:
            raise Exception("Expecting data frame that is indexed 0,1,2,3, ...")
        
        result = [np.nan] * len(X)
        
        # dictionary of sets. The dicitonary has the first columns of the
        # dataframe as key and the set contains all the values of the last column
        dictionary = {}
        
        for row in X.itertuples(index=True):
            index = row[0]
            firstColumns = row[1:-1]
            lastColumn = row[-1]
            
            # Skip firstColumns containing NaN (dictionaries do not work well
            # with np.float(np.NaN) values)
            # Pandas also ignores groups containing NaN
            if any([x != x for x in firstColumns]):
                continue
             
            if not firstColumns in dictionary:
                dictionary[firstColumns] = set()
            
            # Ignore NaN values in last column
            if lastColumn == lastColumn:
                dictionary[firstColumns].add(lastColumn)
            
            result[index] = len(dictionary[firstColumns])
        
        result = pd.DataFrame(result)
        return result

    
class LastTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        result = [np.nan] * len(X)
        
        # dictionary of numbers. The dicitonary has the first columns of the
        # dataframe as key and the current sum for those as value.
        dictionary = {}
        
        for row in X.itertuples(index=True):
            index = row[0]
            firstColumns = row[1:-1]
            lastColumn = row[-1]
            
            # Skip firstColumns containing NaN (dictionaries do not work well
            # with np.float(np.NaN) values)
            # Pandas also ignores groups containing NaN
            if any([x != x for x in firstColumns]):
                continue
            
            if not firstColumns in dictionary:
                dictionary[firstColumns] = "NA"
                
            # this is the last value (e.g., the last property)
            result[index] = dictionary[firstColumns]
            
            # Ignore NaN values in last column
            if lastColumn == lastColumn:
                dictionary[firstColumns] = lastColumn
        
        result = pd.DataFrame(result)
        
        # necessary for firstColumns containing NaN
        result = result.fillna("NA")
          
        return result

    
###############################################################
# Time Transformers
###############################################################
class TimeTransformer(TransformerMixin):
    def __init__(self, unit):
        self._unit = unit
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        timestampSeries = X.iloc[:, 0]
        result = pd.DataFrame()
        
        if self._unit == 'hourOfDay':
            result['hourOfDay'] = pd.DataFrame(timestampSeries.dt.hour)
        elif self._unit == 'dayOfWeek':
            result['dayOfWeek'] = pd.DataFrame(timestampSeries.dt.weekday)
        elif self._unit == 'dayOfMonth':
            result['dayOfMonth'] = pd.DataFrame(timestampSeries.dt.day)
        else:
            raise Exception('undefined unit')
        
        return result

    
#########################################
# Special Transformer
#####################################
class TimeSinceLastRevisionTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    # assumption: first column is username, second column is timestamp
    def transform(self, X):
        X = X.copy()
        
        # convert to seconds since initial point in time
        X['timestamp'] = X['timestamp'].astype(np.int64)
        result = X.groupby('userName').diff(1)
        
        return result

    
##############################################################
# Tag Transformers for Revision Tags
##############################################################
# class FirstTagTransformer(TransformerMixin):
#     """
#     Returns the first tag for each row in a single column.
#     For example, ['def,abc','abc', np.nan] would be transformed to ['def', 'abc', np.nan]
#     """
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         result = X.iloc[:,0].str.split(',').str.get(0)
#
#         result = result.to_frame()
#         return result


class MostFrequentTagTransformer(TransformerMixin):
    """
    Returns the most frequent tag for each row in a single column.
    For example, ['def,abc','abc', np.nan] would be transformed to
    ['abc', 'abc', np.nan] (because abc is more frequent than def).
    """
    
    def fit(self, X, y=None):
        logger.debug("Fitting revisionTag...")
        logger.debug("Splitting...")
        tmp = X.iloc[:,0].str.split(',')
        logger.debug("Dropping...")
        tmp = tmp.dropna()
        logger.debug("Summing up...")
        
        # this line is incredibly slow, hence we use the following alternative
        # tmp = tmp.sum()
        
        tmp = tmp.tolist()
        tmp = list(itertools.chain.from_iterable(tmp))
         
        logger.debug("counting values...")
        self.__freq = pd.Series(tmp).value_counts()
        logger.debug("Fitting revisionTag... done.")
        return self
        
    def transform(self, X):
        logger.debug('Transforming revisionTag...')
        
        def get_freq(element):
            result = self.__freq.get(element)
            if result is None:
                result = 0
            
            return result
        
        def most_frequent(x):
            if x == x:
                result = sorted(x, key=get_freq, reverse=True)[0]
            else:
                result = x  # x is NaN
            return result
        
        mapping = {}
        for category in X.iloc[:, 0].cat.categories:
            mapping[category] = most_frequent(category.split(','))
            
        result = X.iloc[:, 0].map(mapping)
        
        result = result.astype('category')
        result = result.to_frame()
        
        logger.debug('Transforming revisionTag... done.')
        
        return result
            
    
##############################################################
# Random Generators
##############################################################
class NormalGenerator(TransformerMixin):
    def __init__(self, seed):
        self.__seed = seed
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        state = np.random.get_state()
        np.random.seed(seed=self.__seed)
        
        result = pd.DataFrame(np.random.normal(size=len(X)))
        
        np.random.set_state(state)
        
        return result


###############################################################
# Bag-Of-Words Model
###############################################################
def save_vandalism_words(vocabulary, counts, y, vandalismScore):
    
    vandalism = np.where(y == True)[0]
    vandalismCount = np.ravel(counts[vandalism].sum(axis=0))
    totalCount = np.ravel(counts.sum(axis=0))
    vandalismProbability = np.divide(vandalismCount, totalCount)
    
    # print feature probabilities
    inv_vocabulary = {v: k for k, v in vocabulary.items()}
    feature_indices = np.argsort(vandalismScore)[::-1]
    
    vandalism_words = [inv_vocabulary[i] for i in feature_indices]
    vandalismCount = vandalismCount[feature_indices]
    totalCount = totalCount[feature_indices]
    vandalismProbability = vandalismProbability[feature_indices]
    vandalismScore = vandalismScore[feature_indices]
    
    df = pd.DataFrame()
    df['words'] = vandalism_words
    df['vandalismCount'] = vandalismCount
    df['totalCount'] = totalCount
    df['vandalismProb'] = vandalismProbability
    df['vandalismScore'] = vandalismScore
    df.to_csv(config.OUTPUT_PREFIX + "_vandalism_vocabulary.csv")
    
    vandalism_words = [word.encode('ascii', errors='replace').decode()
                       for word in vandalism_words]
    logger.info("top vandalism words:\n" + ", ".join(vandalism_words[:100]))

   
def gridSearch(X, y):
    #     clf = Pipeline([
    #         ('vect', sklearn.feature_extraction.text.CountVectorizer()),
    #         ('tfidf', sklearn.feature_extraction.text.TfidfTransformer()),
    #         ('clf', sklearn.linear_model.SGDClassifier(loss="log")),
    #         #('clf', sklearn.naive_bayes.MultinomialNB()),
    #     ])
    #
    #     parameters = {
    #         'vect__token_pattern': [r"(?u)\b\S+\b", r"(?u)\b\w\w+\b"],
    #         'vect__binary': [True, False],
    #         'vect__min_df': [5],
    #         'tfidf__use_idf': (False,),
    #         'clf__alpha': [10**-6],
    #         #'clf__penalty': ['l2', 'l1'],
    #         #'clf__n_iter': [5,50],
    #         #'clf__average': [False, True]
    #     }
        
    #     clf = sklearn.linear_model.SGDClassifier(loss="log")
    #     parameters = {
    #         'alpha': [10**-i for i in [2, 4, 6, 8, 10]],
    #      }

    clf = sklearn.linear_model.LogisticRegression(solver='sag')
    parameters = {
       'C': [10**-i for i in [-4, -3, -2, -1, 0, 1, 2, 3, 4]],
    }
    
    search = sklearn.grid_search.GridSearchCV(clf, parameters, scoring='roc_auc',
                                              n_jobs=-1, verbose=0)
    search.fit(X, y)
    
    logger.info("All Scores: \n"  + str(search.grid_scores_))
    logger.info("Best Score: "  + str(search.best_score_))
    logger.info("Best Params: " + str(search.best_params_))
    
    return search.best_estimator_


# http://www.win-vector.com/blog/2012/07/modeling-trick-impact-coding-of-categorical-variables-with-many-levels/
class ImpactTransformer(TransformerMixin):
    def __init__(self, alpha):
        self.__alpha = alpha
        
    def fit(self, X, y=None):
        y = X.iloc[:, 1]
        X = X.copy().iloc[:, 0]
        X = X.astype('str')
        
        n = len(X)
        p = sum(y) / n
        
        X = X.append(pd.Series(['__NA'] * len(X)))
        y = y.append(pd.Series(y))
        
        levelcounts = pd.crosstab(X, y, dropna=False)
        
        self.__condprobmodel = \
            (((levelcounts.iloc[:, 1] + p * self.__alpha) / (levelcounts.iloc[:, 0] +
              levelcounts.iloc[:, 1] +
              1 * self.__alpha)))
        self.__condprobmodel.name = "__condprobmodel"
        
        return self
                               
    def transform(self, X):
        X = X.copy().iloc[:, 0]
        X = X.astype('str')
        X = pd.DataFrame(X)
        naval = self.__condprobmodel['__NA']
        
        result = X.join(self.__condprobmodel, on=X.columns[0], how='left')
        result = result['__condprobmodel'].fillna(naval)
        result = result.to_frame()

        return result
        

class BadWordRatioTransformer(TransformerMixin):
    def __init__(self, minWords, threshold):
        self.__minWords = minWords
        self.__threshold = threshold
    
    def fit(self, X, y=None):
        logger.debug("BadWordRatioTransformer fitting...")
        y = X.iloc[:, 1]
        X = X.copy().iloc[:, 0]
        X = X.astype('str')
        
        logger.debug("Vectorizing...")
        vectorizer = (sklearn.feature_extraction.text
                      .CountVectorizer(min_df=self.__minWords))
        X = vectorizer.fit_transform(X)
        logger.debug("Vectorizing...done")
               
        vandalism = np.where(y)[0]
        vandalismSum = np.ravel(X[vandalism].sum(axis=0))
        totalSum = np.ravel(X.sum(axis=0))
        vandalismProbability = np.divide(vandalismSum, totalSum)
        
        inv_vocabulary = {v: k for k, v in vectorizer.vocabulary_.items()}
        vocabulary_list = np.array(
            [inv_vocabulary[i] for i in range(len(inv_vocabulary))])
        self.__badWords = vocabulary_list[
            np.array(vandalismProbability > self.__threshold)]
        
        save_vandalism_words(vectorizer.vocabulary_, X, y, vandalismProbability)
               
        return self
    
    def transform(self, X):
        logger.debug("BadWordRatioTransformer transforming...")
        # y= X.iloc[: ,1]
        X = X.copy().iloc[:, 0]
        X = X.astype('str')
        
        logger.debug("counting bad words...")
        if len(self.__badWords) > 0:
            badWordVectorizer = sklearn.feature_extraction.text.CountVectorizer(
                min_df=self.__minWords, vocabulary=self.__badWords)
            badWords = badWordVectorizer.fit_transform(X)
            nBadWords = badWords.sum(axis=1)
        else:
            nBadWords = scipy.sparse.csr_matrix(
                np.zeros((len(X), 1), dtype=np.int64)).sum(axis=1)
        
        logger.debug("counting all words...")
        allWordVectorizer = sklearn.feature_extraction.text.CountVectorizer()
        allWords = allWordVectorizer.fit_transform(X)
        nAllWords = allWords.sum(axis=1)
        
        result = np.divide(nBadWords, nAllWords)
        result = np.array(result).ravel()
     
        result = pd.DataFrame(result)
        result = result.astype(np.float32)
        return result
    

class CommentTransformer(TransformerMixin):
    def fit(self, X, y=None):
        logger.debug("CommentTransformer fitting...")
        y = X.iloc[:, 1]
        X = X.copy().iloc[:, 0]
        X = X.astype('str')
        
        # self.__vectorizer =sklearn.feature_extraction.text.CountVectorizer(
        #    analyzer='char', min_df=100, ngram_range=(2,3))
        self.__vectorizer = sklearn.feature_extraction.text.CountVectorizer(min_df=5)
       
        X = self.__vectorizer.fit_transform(X)
        # self.__clf = gridSearch(X,y )
        # self.__clf = sklearn.linear_model.SGDClassifier(loss="log", alpha=10**-6)
        self.__clf = sklearn.linear_model.LogisticRegression(
            solver='sag', C=1.0, random_state=1, verbose=0, n_jobs=-1)
        
        self.__clf.fit(X, y)
        
        save_vandalism_words(self.__vectorizer.vocabulary_, X, y, self.__clf.coef_[0])
               
        return self
    
    def transform(self, X):
        logger.debug("CommentTransformer transforming...")
        X = X.copy().iloc[:, 0]
        X = X.astype('str')
        X = self.__vectorizer.transform(X)
   
        result = self.__clf.predict_proba(X)[:, 1]
        
        # because the values can be very small and it is later converted to float32
        result = np.log2(result + 10**-12)
        result = np.round(result)  # to avoid overfitting
        result = pd.DataFrame(result)
        result = result.astype(np.float32)
                
        return result

    
class CommentTransformerOld(TransformerMixin):
    def __init__(self):
        from sklearn.feature_extraction.text import CountVectorizer
        # character analyzer takes very long
        # self.__vectorizer = CountVectorizer(
        #    analyzer='char', binary=True, min_df=100,
        #    token_pattern=r"(?u)\b\S+\b", ngram_range=(2,3))
        self.__vectorizer = CountVectorizer(
            analyzer='word', binary=True, min_df=5,
            token_pattern=r"(?u)\b\S+\b", ngram_range=(1, 1))
    
    def fit(self, X, y=None):
        logger.debug("CommentTransformer fitting...")
        tmp = X.copy().iloc[:, 0]
        tmp = tmp.astype('str')
        
        logger.debug("Counting...")
        counts = self.__vectorizer.fit_transform(tmp)
        logger.debug("Counting...done")
               
        vandalism = np.where(X.iloc[:, 1] == True)[0]
        vandalismSum = np.ravel(counts[vandalism].sum(axis=0))
        totalSum = np.ravel(counts.sum(axis=0))
        vandalismProbability = np.divide(vandalismSum, totalSum)
        
        # Laplacian smoothing (to disable it, set alpha = 0)
#         alpha = 0.01
#         wordProbVand = ((vandalismSum + alpha) /
#             (len(vandalism) + (alpha * len(vandalismSum))))
#         wordProb = totalSum / len(X)
#         self.__vandalismPrior = len(vandalism) / len(X)

        # if alpha == 0, this is simply vandalismSum / totalSum
#         theta = self.__vandalismPrior * np.divide(wordProbVand, wordProb)

        # Alternative smoothing
#         mu = 100
#         theta = np.divide(
#              vandalismSum + (mu / len(vandalismSum)), totalSum + mu)

        theta = vandalismProbability
        
        save_vandalism_words(self.__vectorizer.vocabulary_, vandalismSum,
                             totalSum, vandalismProbability, theta)
        
        self.__theta = scipy.sparse.diags([theta], [0])  # main diagonal
 
        return self
    
    def transform(self, X):
        logger.debug("CommentTransformer transforming...")
        tmp = X.copy().iloc[:, 0]
        tmp = tmp.astype('str')
                
        logger.debug("counting...")
        counts = self.__vectorizer.transform(tmp)
        logger.debug("multiplying...")
        rowTheta = counts * self.__theta
        logger.debug("multiplying...done")
        
        # only consider suffix comments consisting of one word
        # rowTheta[rowTheta.sum(axis=1) > 1] = 0
        
        rowAggTheta = rowTheta.max(axis=1).toarray()
        
        # rowAggTheta = rowTheta.mean(axis=1) # max seems to be better
        
        # Imputing missing values, self.__vandalismPrior is problematic due to
        # maximum and due to smoothing
        rowAggTheta[rowAggTheta == 0] = rowAggTheta[rowAggTheta > 0].mean()
        rowAggTheta = np.ravel(rowAggTheta)
        
        logger.debug(
            "CommentTransformer AUC-ROC: " +
            str(sklearn.metrics.roc_auc_score(X.iloc[:, 1], rowAggTheta)))
        logger.debug(
            "CommentTransformer AUC-PR: " +
            str(evaluation.goadrich_pr_auc_score(X.iloc[:, 1], rowAggTheta)))
        
        result = pd.DataFrame(rowAggTheta)
        # result = result.apply(np.log2)
        # result = result.apply(np.round)
        
        return result
          

###############################################################################
# Unused Transformers (because they are too slow)
###############################################################################
class CumUniqueTransformerPandas(TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        firstColumns = list(X.columns[0:-1])
        # lastColumn = X.columns[-1]
        
        logger.debug("dropping duplicates...")
        without_duplicates = X.drop_duplicates()
        
        logger.debug("grouping...")
        unique_count = (without_duplicates.groupby(firstColumns)
                                          .cumcount()
                                          .astype(np.float32) + 1)
        unique_count.name = 'unique_count'
        
        logger.debug("joining...")
        result = X.join(unique_count, how='left')
        
        logger.debug("grouping...")
        grouped = result.groupby(firstColumns)
        
        # very slow (because internall it does not use cython and creates
        # a lot of data frames)
        logger.debug("filling...")
        result = grouped.ffill()
        
#         import timeit
#         logger.debug(str(timeit.timeit(lambda: test(grouped), number=1)))
#         logger.debug(str(timeit.timeit(grouped.ffill, number=1)))
        
        logger.debug("transforming to frame...")
        result = result[unique_count.name].to_frame()
        
        return result
