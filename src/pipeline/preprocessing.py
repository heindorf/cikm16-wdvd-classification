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

import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed

import config
from .. import featurelist
from .. import transformers
from .. import utils
from ..dataset import DataSet
from ..utils import Timer

_logger = logging.getLogger()


# Given a pandas data frame and a list of features,
# applies the features' transformers on the data frame and
# returns the whole transformed data set
def fit_transform(time_label, system_name, data, features, fit_slice):
    _logger.info("Preprocessing...")
    utils.collect_garbage()
      
    data = _applyTransformers(data, features, fit_slice)
     
    _checkTransformation(data.getX(), data.getFeatureNames())
    
    utils.collect_garbage()
    
    _logger.info("Preprocessing... done.")
    
    data.setTimeLabel(time_label)
    data.setSystemName(system_name)
    
    return data


###############################################################
# Applying all transformers (including imputers for missing values)
###############################################################
def _applyTransformers(data, feature_list, fit_slice):
    _logger.debug("Starting transformation...")

    # For actual features, adds a Float32Transformer (--> less data to transfer
    # back across processes)
    # Meta features are not transformed to float32.
    data_list = feature_list + featurelist.get_label_list()
    for feature in data_list:
        feature.getTransformers().append(transformers.Float32Transformer())
    
    feature_list = featurelist.get_meta_list() + data_list
    
    arguments = []
    
    for feature in feature_list:
        featureInputNames = feature.getInputNames()
        if featureInputNames:  # list is not empty
            curData = data.loc[:, featureInputNames]
        else:
            curData = pd.DataFrame()
             
        arguments.append((feature, curData, fit_slice, ))
        
    # this is the central part of this function
    column_list = Parallel(n_jobs=config.PREPROCESSING_N_JOBS, backend='multiprocessing')(
        delayed(_process_feature_parallel)(*arguments[i]) for i in range(len(arguments)))
       
    n_meta = len(featurelist.get_meta_list())
    newMeta = pd.concat(column_list[0:n_meta], axis=1)
    
    data_columns = column_list[n_meta:-1]
    
    # We use our own "combine function" because pd.concat needs a lot of memory
    newX, feature_list = _combine_columns(data_columns)
        
    newY = column_list[-1].iloc[:, 0].astype(np.float32).values

    utils.collect_garbage()
    
    newData = DataSet()
    newData.setMeta(newMeta)
    newData.setX(newX)
    newData.setY(newY)
    newData.setFeatures(feature_list)
    
    return newData


def _checkTransformation(training_X, feature_names):
    if (len(feature_names) != training_X.shape[1]):
        raise Exception("There are " +
                        str(len(feature_names)) + " feature names but " +
                        str(training_X.shape[1]) + " features in the data")


def _generateColumn(feature, name):
    if (feature.getGroup() != None) & (feature.getSubgroup() != None):
        result = pd.MultiIndex.from_tuples(
            [(feature.getGroup(), feature.getSubgroup(), name)],
            names=['group', 'subgroup', 'feature'])
    else:
        result = [name]
    
    return result


# This method is called by multiple processes
def _process_feature_parallel(feature, curData, fit_slice):
    _logger.debug("Starting feature " + str(feature) + "...")
    utils.collect_garbage()
    
    with Timer() as t:
        transformers = feature.getTransformers()
        
        for transformer in transformers:
            _logger.debug(str(transformer))
            try:
                fittingData = curData[fit_slice]
                
                curData = transformer.fit(fittingData).transform(curData)
                
                if not isinstance(curData, pd.DataFrame):
                    raise Exception("return type should be pandas.DataFrame but was " +
                                    str(type(curData)))
 
            except:
                _logger.error(feature.getOutputName() + ": " + str(transformer))
                raise
            
        if len(curData.columns.values) == 1:
            name = feature.getOutputName()
            curData.columns = _generateColumn(feature, name)
        
        elif len(curData.columns.values) > 1:
            newColumns = []
            for i in range(len(curData.columns.values)):
                name = feature.getOutputName() + "_" + str(curData.columns.values[i])
                column = _generateColumn(feature, name)
                newColumns.append(column)
            curData.columns = newColumns
        
        _logger.debug("concatenating...")
        
    _logger.debug(
        "=> elapsed time for feature %s: %d s (current data size: %d MB)"
        % (feature.getOutputName(), t.secs, curData.memory_usage().sum() / 1024 / 1024))
    
    utils.collect_garbage()
    
    return curData


# returns np.array
def _combine_columns(df_list):
    result = pd.DataFrame()
    feature_list = []
    
    for df in df_list:
        for column in df.columns:
            if column in result.columns:
                raise Exception("duplicate column: " + str(column))
            
            if df[column].dtype != "float32":
                _logger.warn("not float32, column: " + column)
            
            result[column] = df[column].astype(np.float32)
            feature_list.append(column)
            
    result = result.values
    return result, feature_list
