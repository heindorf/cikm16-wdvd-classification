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
import collections
import pprint

import pandas as pd
import sklearn.ensemble
import sklearn.grid_search
import sklearn.base
from sklearn.ensemble import RandomForestClassifier


import config
from src import evaluation
from sklearn.externals.joblib import Parallel, delayed

_logger = logging.getLogger()


def getDefaultRandomForest():
    # We explicitly set the initial random_state to ensure reproducibility
    clf = RandomForestClassifier(n_jobs=-1, verbose=0, random_state=1)
    return clf

_CLASSIFIER = {
    'FOREST': getDefaultRandomForest(),
    'BAGGING': sklearn.ensemble.BaggingClassifier(
        base_estimator=getDefaultRandomForest(),
        n_jobs=1, verbose=0, random_state=1),
    'SAMPLING': sklearn.ensemble.BaggingClassifier(
        base_estimator=getDefaultRandomForest(),
        n_estimators=1, n_jobs=1),
}

_PARAM_GRID = {}
_PARAM_GRID['FOREST'] = {
    'n_estimators': [10],
    'max_features': [1, 2, 'log2', 'sqrt'],
    'max_depth': [1, 2, 4, 8, 16, 32, 64, None]
}

_PARAM_GRID['BAGGING'] = {
    'base_estimator__n_estimators': [8],
    'base_estimator__max_features': [1, 2, 'log2', 'sqrt'],
    'base_estimator__max_depth': [1, 2, 4, 8, 16, 32, 64, None],
    'n_estimators': [16],
    'max_samples': [1 / 16]
}

_PARAM_GRID['SAMPLING'] = {
    'max_samples': [1 / (2**i) for i in range(11)],
    'random_state': range(10)
}


def _get_params_list(classifier_name):
    result = []
    
    if (classifier_name == 'SAMPLING'):
        for params in sklearn.grid_search.ParameterGrid(_PARAM_GRID[classifier_name]):
            # add randomization to underlying base estimator
            params['base_estimator__random_state'] = params['random_state'] + 1000
            result.append(params)
        
    elif (classifier_name == 'FOREST') | (classifier_name == 'BAGGING'):
        for params in sklearn.grid_search.ParameterGrid(_PARAM_GRID[classifier_name]):
            result.append(params)
    else:
        raise Exception("unknown classifer name: " + classifier_name)
    
    return result

# Cached results. TODO: update optimal params if PARAM_GRID has changed.
_OPTIMAL_PARAMS = {'WDVD': {},
                   'FILTER': {},
                   'ORES': {}}


_OPTIMAL_PARAMS = {
 'WDVD':   {'FOREST':  {'max_depth': 8,
                        'max_features': 'sqrt',
                        'n_estimators': 10},
            'BAGGING': {'base_estimator__max_depth': 32,
                        'base_estimator__max_features': 2,
                        'base_estimator__n_estimators': 8,
                        'max_samples': 1 / 16,
                        'n_estimators': 16}},
 'FILTER': {'FOREST':  {'max_depth': 16,
                        'max_features': 1,
                        'n_estimators': 10},
            'BAGGING': {'base_estimator__max_depth': 8,
                        'base_estimator__max_features': 1,
                        'base_estimator__n_estimators': 8,
                        'max_samples': 1 / 16,
                        'n_estimators': 16}},
 'ORES':   {'FOREST':  {'max_depth': 16,
                        'max_features': 'sqrt',
                        'n_estimators': 10},
            'BAGGING': {'base_estimator__max_depth': 16,
                        'base_estimator__max_features': 'sqrt',
                        'base_estimator__n_estimators': 8,
                        'max_samples': 1 / 16,
                        'n_estimators': 16}}}

_RESULTS = {
    'FOREST': pd.DataFrame(),
    'BAGGING': pd.DataFrame(),
    'SAMPLING': pd.DataFrame(),
}


def optimize(training, validation):
    if config.OPTIMIZATION_ENABLED:
        _logger.info("Optimizing...")
    
        _OPTIMAL_PARAMS[validation.getSystemName()] = {}

        _optimize('FOREST', training, validation)
        _optimize('BAGGING', training, validation)
        _optimize('SAMPLING', training, validation)
        
        _logger.info("Optimal Parameters: " + str(_OPTIMAL_PARAMS))
        file = open(config.OUTPUT_PREFIX + '_optimal_params.py', 'w')
        pprint.pprint(_OPTIMAL_PARAMS, file)
    
        _logger.info("Optimizing... done.")
    

def getOptimalRandomForest(system_name):
    clf = sklearn.base.clone(_CLASSIFIER['FOREST'])
    clf.set_params(**_OPTIMAL_PARAMS[system_name]['FOREST'])
    
    return clf

  
def getOptimalBaggingClassifier(system_name):
    clf = sklearn.base.clone(_CLASSIFIER['BAGGING'])
    clf.set_params(**_OPTIMAL_PARAMS[system_name]['BAGGING'])
    return clf


####################################################
# Optimize Random Forest
####################################################
def _optimize(classifier_name, training, validation):
    _logger.info('Optimizing ' + classifier_name + '...')
    
    params_list = _get_params_list(classifier_name)
    
    # remove invalid parameter combinations
    if validation.getNumberOfFeatures() < 2:
        new_param_list = []
        for params in params_list:
            if 'max_features' in params and params['max_features'] == 2:
                pass
            elif ('base_estimator__max_features' in params and
                  params['base_estimator__max_features'] == 2):
                pass
            else:
                new_param_list.append(params)
        params_list = new_param_list
    
    clf = _CLASSIFIER[classifier_name]
    metrics, optimal_params = _grid_search(
        classifier_name, clf, params_list, training, validation)
    
    _RESULTS[classifier_name] = _RESULTS[classifier_name].append(metrics)
    evaluation.print_metrics(
        _RESULTS[classifier_name], 'OPTIMIZATION_' + classifier_name, False)
    
    _OPTIMAL_PARAMS[validation.getSystemName()][classifier_name] = optimal_params
    
    _logger.info('Optimizing ' + classifier_name + '... done.')


############################################################
# Grid Search
############################################################
def _convert_to_index(system_name, classifier_name, params):
    index = collections.OrderedDict()
    
    index['System'] = system_name
    index['Classifier'] = classifier_name
    index.update(params)

    index = pd.MultiIndex.from_tuples([tuple(index.values())], names=index.keys())

    return index


def _convert_from_index(multiindex):
    params = dict(zip(multiindex.names, multiindex.values[0]))
    params.pop('System', None)
    params.pop('Classifier', None)
               
    return params


def _grid_search(classifier_name, clf, params_list, training, validation):
    _logger.debug("grid search...")

    arguments = []
    for params in params_list:
        index = _convert_to_index(validation.getSystemName(), classifier_name, params)
        argument = (index, clf, params, training, validation)
        arguments.append(argument)
        
    metrics = Parallel(n_jobs=config.OPTIMIZATION_N_JOBS, backend='multiprocessing')(
        delayed(_grid_search_parallel)(*x) for x in arguments)
    
    # insert dummy to make sure all columns have the correct dtype ('object')
    dummy = tuple(['dummy'] * metrics[0].index.nlevels)
    index = pd.MultiIndex.from_tuples([dummy])
    dummy_df = metrics[0].copy()
    dummy_df.index = index
    metrics.append(dummy_df)
        
    metrics = pd.concat(metrics, axis=0)
    
    # metrics = metrics.drop(dummy) # drop dummy again
    
    pr_column = metrics.loc[:, ('ALL', 'PR')]
    idxmax = pr_column.idxmax()
    
    optimum = metrics.loc[[idxmax], :]
    
    optimal_params = _convert_from_index(optimum.index)
    
    _logger.debug("grid search... done.")
    
    return metrics, optimal_params
    
    
def _grid_search_parallel(index, clf, params, training, validation):
    # sort params for printing
    params = collections.OrderedDict(sorted(params.items()))
     
    clf = sklearn.base.clone(clf)
    clf.set_params(**params)
    
    try:
        _, _, metrics = evaluation.fit_predict_evaluate(index, clf, training, validation)
        
    except Exception as e:
        _logger.warn("Could not perform grid search for params " +
                     str(params) + " Exception: " + str(e))
            
    return metrics
