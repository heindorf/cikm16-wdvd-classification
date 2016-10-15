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
import pandas as pd

import config

from . import optimization
from .. import evaluation
from ..classifiers import multipleinstance

_logger = logging.getLogger()

_metrics = {}


# prefix can, for example, be 'validation' or 'test'
def classify(training, validation, classify_groups=False):
    _logger.info("Classifying...")
    
    default_random_forest(training, validation)
    optimized_random_forest(training, validation)
    bagging_and_multiple_instance(training, validation)
    
    if classify_groups:
        _compute_metrics_for_classifiers_and_groups(training, validation)
    
    _logger.info("Classifying... done.")
    
  
def default_random_forest(training, validation, print_results=True):
    _logger.info("Default random forest...")
    clf = optimization.getDefaultRandomForest()
    clf.set_params(n_jobs=config.CLASSIFICATION_N_JOBS)
    
    index = _getIndex(validation.getSystemName(), 'Default random forest')
    _, _, metrics = evaluation.fit_predict_evaluate(
        index, clf, training, validation)
    
    if print_results:
        _print_metrics(validation.getTimeLabel(), metrics)
    _logger.info("Default random forest... done.")
    
    return metrics
    
    
def _getIndex(system_name, classifier_name):
    
    result = pd.MultiIndex.from_tuples(
        [(system_name, classifier_name)], names=['System', 'Classifier'])
    return result
    
        
def optimized_random_forest(training, validation, print_results=True):
    _logger.info("Optimized random forest...")
    clf = optimization.getOptimalRandomForest(validation.getSystemName())
    clf.set_params(n_jobs=config.CLASSIFICATION_N_JOBS)
      
    index = _getIndex(validation.getSystemName(), 'Optimized random forest')
    _, _, metrics = evaluation.fit_predict_evaluate(
        index, clf, training, validation)
    if print_results:
        _print_metrics(validation.getTimeLabel(), metrics)
    _logger.info("Optimized random forest... done.")
    
    return metrics
    
    
def bagging_and_multiple_instance(training, validation, print_results=True):
    _logger.info("Bagging and multiple-instance...")
    
    result = pd.DataFrame()
    
    # Bagging
    clf = optimization.getOptimalBaggingClassifier(validation.getSystemName())
    
    # for optimization another number of jobs is used (possibly)
    clf.set_params(n_jobs=config.CLASSIFICATION_N_JOBS)
    
    index = _getIndex(validation.getSystemName(), 'Bagging')
    _, prob, metrics = evaluation.fit_predict_evaluate(
        index, clf, training, validation)
    result = result.append(metrics)
    if print_results:
        _print_metrics(validation.getTimeLabel(), metrics)
    
    # Single-instance learning (SIL)
    clf = multipleinstance.SingleInstanceClassifier(base_estimator=None)
    clf.set_proba(prob)  # shortcut to save some computational time
    index = _getIndex(validation.getSystemName(), 'SIL MI')
    sil_pred, sil_prob = evaluation.predict(clf, validation, index)
    metrics = evaluation.evaluate(index, sil_pred, sil_prob, validation)
    result = result.append(metrics)
    if print_results:
        _print_metrics(validation.getTimeLabel(), metrics)
    
    # Simple multiple-instance (SMI)
    clf = optimization.getOptimalBaggingClassifier(validation.getSystemName())
    
    # for optimization another number of jobs is used (possibly)
    clf.set_params(n_jobs=config.CLASSIFICATION_N_JOBS)
    
    clf = multipleinstance.SimpleMultipleInstanceClassifier(base_estimator=clf)
    index = _getIndex(validation.getSystemName(), 'Simple MI')
    _, smi_prob, metrics = evaluation.fit_predict_evaluate(
        index, clf, training, validation)
    result = result.append(metrics)
    if print_results:
        _print_metrics(validation.getTimeLabel(), metrics)
    
    # Combination of SIL and SMI
    clf = multipleinstance.CombinedMultipleInstanceClassifier(
        base_estimator=None)
    # shortcut to save some computational time
    clf.set_proba(sil_prob, smi_prob)
    index = _getIndex(validation.getSystemName(), 'Combined MI')
    combined_pred, combined_prob = evaluation.predict(clf, validation, index)
    metrics = evaluation.evaluate(
        index, combined_pred, combined_prob, validation)

    result = result.append(metrics)
    if print_results:
        _print_metrics(validation.getTimeLabel(), metrics)
        
    _logger.info("Bagging and multiple-instance... done.")
        
    return result


def _compute_metrics_for_classifiers_and_groups(training, validation):
    arguments = []
    for clf_name in ['DFOREST', 'OFOREST', 'BAGGING']:
        argument = (training, validation, 'ALL', clf_name)
        arguments.append(argument)
        
        for group in validation.getGroups():
            training2 = training.select_group(group)
            validation2 = validation.select_group(group)
            argument = tuple([training2, validation2, group, clf_name])
            arguments.append(argument)
            
    local_metrics = pd.DataFrame()
            
    for argument in arguments:
        result = _compute_metrics_for_classifier(*argument)
        local_metrics = local_metrics.append(result)
        local_metrics.to_csv(config.OUTPUT_PREFIX + '_' +
                             validation.getTimeLabel() +
                             '_classifiers_groups.csv')
        screen_output = evaluation.remove_plots(local_metrics)
        _logger.info("Metrics:\n" + str(screen_output))
            

def _compute_metrics_for_classifier(training, validation, group, clf_name):
    if clf_name == 'DFOREST':
        result = default_random_forest(training, validation, False)
    elif clf_name == 'OFOREST':
        result = optimized_random_forest(training, validation, False)
    elif clf_name == 'BAGGING':
        result = bagging_and_multiple_instance(training, validation, False)
        
    result['Group'] = group
    result.set_index('Group', append=True, inplace=True)
    
    return result
        

def _print_metrics(time_label, metrics):
    global _metrics
    if time_label not in _metrics:
        _metrics[time_label] = pd.DataFrame()
        
    _metrics[time_label] = _metrics[time_label].append(metrics)
    
    local_metrics = _metrics[time_label].copy()
    
    local_metrics = reverse_order_within_system_groups(local_metrics)
    
    local_metrics.to_csv(
        config.OUTPUT_PREFIX + '_' + time_label + '_results.csv')
    
    local_metrics = evaluation.remove_plots(local_metrics)
    _logger.info("Metrics:\n" + str(local_metrics))
    evaluation.print_metrics_to_latex(local_metrics,
                                      config.OUTPUT_PREFIX + '_' +
                                      time_label + '_results.tex')

  
# prints misclassified examples and determines threhsold according to fpr
def _print_examples(validation, y_score, threshold):
    meta = validation.getMeta().copy()
    meta['y_true'] = validation.getY()
    meta['y_score'] = y_score
    meta['y_pred'] = meta['y_score'] > threshold
    
    # divide by content type
    head_revisions = meta[meta['contentType'] == 'TEXT']
    statement_revisions = meta[meta['contentType'] == 'STATEMENT']
    sitelink_revisions = meta[meta['contentType'] == 'SITELINK']
    
    _logger.info("Threshold: %f" % threshold)
    _print_cell(validation.getTimeLabel(),
                'head revisions', head_revisions)
    _print_cell(validation.getTimeLabel(),
                'statement revisions', statement_revisions)
    _print_cell(validation.getTimeLabel(),
                'sitelink revisions', sitelink_revisions)


def _print_cell(timelabel, label, df):
    _logger.info('Number of %s which are vandalism: %d' %
                 (label, df['y_true'].sum()))
    _logger.info('Number of %s which are regular: %d' %
                 (label, len(df) - df['y_true'].sum()))
    
    N_SAMPLES = 100
    n = min(len(df), N_SAMPLES)
    if n > 0:
        # balance vandalism/regular edits
        weights = df.groupby(by='y_true')['y_true'].transform(
            lambda x: (1 / 2) / x.count())
        df = df.sample(n, weights=weights, random_state=1)
    df.to_csv(config.OUTPUT_PREFIX + '_' + timelabel + '_' +
              label.replace(" ", "_") + '.csv', index=False)
  
    
def reverse_order_within_system_groups(metrics):
    # The following line does not work (bug in Pandas?)
    # local_metrics = local_metrics.groupby(level='System', sort=False).apply(
    #     lambda x: x.reindex(index=x.index[::-1])
    
    # returns a list of tuples based on the data frame's multiindex
    old_order = metrics.index.values
    new_order = []
    
    for _, group in itertools.groupby(old_order, lambda x: x[0]):
        new_order = new_order + list(group)[::-1]
    
    result = metrics.reindex(new_order)
    
    return result
