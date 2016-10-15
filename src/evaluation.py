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

import collections
import csv
import itertools
import logging
import os
import re
import subprocess
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import sklearn.metrics
from sklearn.metrics import precision_recall_curve

import config
from . import utils
from src.classifiers import multipleinstance

COLUMN_LIST = list(itertools.product(['ITEM_HEAD', 'ITEM_BODY', 'ALL'], ['ROC', 'PR']))

_metrics = pd.DataFrame()
_logger = logging.getLogger()


#################################################################
# Computing metrics of classifiers / features
#################################################################
def fit(clf, dataset, index=''):
    label = _get_label(index)
    
    _logger.debug("Fitting %s..." % label)
    if (isinstance(clf, multipleinstance.BaseMultipleInstanceClassifier)):
        clf.fit(dataset.getGroupIDs().values, dataset.getX(), dataset.getY())
    else:
        clf.fit(dataset.getX(), dataset.getY())
       
    _logger.debug("Fitting %s... done." % label)

   
def predict(clf, dataset, index=''):
    label = _get_label(index)
    
    _logger.debug("Predicting %s..." % label)
    
    if (isinstance(clf, multipleinstance.BaseMultipleInstanceClassifier)):
        # second column denotes the probability for vandalism
        prob = clf.predict_proba(dataset.getGroupIDs().values, dataset.getX())
    else:
        # second column denotes the probability for vandalism
        prob = clf.predict_proba(dataset.getX())[:, 1]
    
    pred = prob > 0.5
    
    _logger.debug("Predicting %s... done." % label)
    return pred, prob


def _get_label(index):
    label = str(index)
    try:
        label = "%s" % (str(index.values[0]))
    except:
        pass
    
    return label


def evaluate_print(name, pred, prob, dataset):
    metrics = evaluate(name, pred, prob, dataset)
    print_metrics(metrics)
    result = metrics.loc[0, ('ALL', 'PR')]
    
    # ROC Curve
    # if( not (prob is None)):
    #    _draw_ROC_plot(dataset.getY(), prob, name)
 
    # PR AUC (using Goadrichs JAR file)
    # if (not (prob is None)):
    #    _draw_AUC_PR_plot(dataset.getY(), prob, name)
     
    # print_some_misclassified_examples(actual, pred, ids)
    # print_most_severely_misclassified_examples(actual, prob, ids)
    # print_random_misclassified_examples(actual, pred, prob, ids)
    # print_random_misclassified_examples_by_content_type(actual, pred, prob, ids, contentTypes, 'STATEMENT')
     
    return result


# In addition to the public evaluate, also stores some benchmarking times
def evaluate(index, pred, prob, dataset, fit_time=-1, prob_time=-1):
    label = _get_label(index)
    _logger.debug("Evaluating %s..." % label)
    
    local_metrics = compute_feature_metrics_for_all_content_types(
        index, dataset.getContentTypes(), dataset.getY(), prob, pred)
    
    idx = local_metrics.columns.get_loc(('ALL', 'PR')) + 1
    
    local_metrics.insert(idx, ('ALL', 'TOTAL_TIME'), fit_time + prob_time)
    local_metrics.insert(idx, ('ALL', 'PROB_TIME'), prob_time)
    local_metrics.insert(idx, ('ALL', 'FIT_TIME'), fit_time)
    
    _logger.debug("Evaluating %s... done." % label)
    
    return local_metrics


def fit_predict_evaluate(index, clf, training, validation):
    fit_start = time.time()
    fit(clf, training, index)
    fit_end = time.time()
    fit_time = fit_end - fit_start

    prob_start = time.time()
    pred, prob = predict(clf, validation, index)

    prob_end = time.time()
    prob_time = prob_end - prob_start
    
    metrics = evaluate(index, pred, prob, validation, fit_time, prob_time)
    
    return pred, prob, metrics


# Fits the training data, predicts the validation data and evaluates the
# classifier clf. The result is a single row of a pandas data frame.
# The classifier clf must implement both predict and predict_from_proba
def fit_predict_evaluate_print(name, clf, training, validation):
    pred, prob, metrics = fit_predict_evaluate(name, clf, training, validation)
    
    print_metrics(metrics)
    
    return pred, prob, metrics


def remove_plots(metrics):
    labels_to_drop = list(itertools.product(
        metrics.columns.levels[0],
        ['recallValues', 'precisionValues',
         'fprValues', 'tprValues', 'rocThresholds']))
    result = metrics.drop(labels_to_drop, axis=1, errors='ignore')
    
    return result


def _remove_duplicates(seq):
    result = []
    for e in seq:
        if e not in result:
            result.append(e)
    return result


def print_metrics(metrics, suffix='metrics', append_global=True):
    if (append_global):
        global _metrics
        
        _metrics = _metrics.append(metrics)
        _print_metrics(_metrics, suffix)
    else:
        _print_metrics(metrics, suffix)

    
def _print_metrics(metrics, suffix):
    metrics.to_csv(config.OUTPUT_PREFIX + '_' + suffix + '.csv')
    
    metrics = remove_plots(metrics)
    
    _logger.info("Metrics:\n" +
                 (metrics.to_string(float_format='{:.4f}'.format)))
    
    print_metrics_to_latex(metrics,
                           config.OUTPUT_PREFIX + '_' + suffix + '.tex')


# prints metrics to latex and formats them as \\bscellA
def print_metrics_to_latex(metrics, filename):
    latex = metrics.loc[:, COLUMN_LIST].copy()
    
    def float_format2(value, char):
        return '\\bscell%s[%.3f]{%3.0f}' % (char, value, value * 100)

    def float_formatA(value):
        return float_format2(value, 'A')

    def float_formatB(value):
        return float_format2(value, 'B')

    # use formatB for all 'ROC' columns
    formatters = {value: float_formatB if value[1] == 'ROC' else None
                  for value in latex.columns.values}
    
    # workaround because the index is not properly formatted in Latex
    latex = latex.reset_index()
 
    latex.to_latex(filename,
                   float_format=float_formatA,
                   formatters=formatters,
                   escape=False, index=False)

    
# This method is called by multiple processes
def compute_feature_metrics_for_all_content_types(
        index, content_types, y_true, y_score, y_pred):
    _logger.debug("Computing feature metrics...")
    utils.collect_garbage()
    
    result              = collections.OrderedDict()
    result['ALL']       = compute_feature_metrics_for_content_type(index, content_types, 'ALL'      , y_true, y_score, y_pred)
    result['ITEM_HEAD'] = compute_feature_metrics_for_content_type(index, content_types, 'ITEM_HEAD', y_true, y_score, y_pred)
    result['ITEM_BODY'] = compute_feature_metrics_for_content_type(index, content_types, 'ITEM_BODY', y_true, y_score, y_pred)
    # result['STATEMENT'] = compute_feature_metrics_for_content_type(index, content_types, 'STATEMENT', y_true, y_score, y_pred)
    # result['SITELINK']  = compute_feature_metrics_for_content_type(index, content_types, 'SITELINK' , y_true, y_score, y_pred)
    
    result = pd.concat(result.values(), axis=1, keys=result.keys())
    
    utils.collect_garbage()
    
    _logger.debug("Computing feature metrics...done.")

    return result


def compute_feature_metrics_for_content_type(
        index, content_types, content_type, y_true, y_score, y_pred):
    if content_type == 'ALL':
        content_type_select = np.ones((len(content_types),), dtype=bool)
    elif content_type == 'ITEM_HEAD':
        content_type_select = np.array(content_types.values == 'TEXT')
    elif content_type == 'ITEM_BODY':
        content_type_select = np.array((content_types.values == 'STATEMENT') |
                                       (content_types.values == 'SITELINK'))
    else:
        content_type_select = np.array(content_types.values == content_type)
        
    y_true = y_true[content_type_select]
    y_pred = y_pred[content_type_select]
    
    if y_score is not None:
        y_score = y_score[content_type_select]
    
    result = collections.OrderedDict()
    
    if len(y_true) > 0 and sum(y_true) > 0:
        # Metrics based on prediction
        
        # logger.debug('ACC...')
        # result['ACC'] = sklearn.metrics.accuracy_score(y_true, y_pred)
        # logger.debug('P...')
        # result['P'] = sklearn.metrics.precision_score(y_true, y_pred)
        # logger.debug('R...')
        # result['R'] = sklearn.metrics.recall_score(y_true, y_pred)
        # logger.debug('F...')
        # result['F'] = sklearn.metrics.f1_score(y_true, y_pred)
 
        # Metrics based on probabilistic score
        if y_score is not None:
            # logger.debug('ROC...')
            result['ROC'] = sklearn.metrics.roc_auc_score(y_true, y_score)
            fpr, tpr, roc_thresholds = _roc_curve(y_true, y_score)
            # logger.debug('PR...')
            precisionValues, recallValues, auc_pr = \
                _goadrich_precision_recall_curve(y_true, y_score)
            result['PR'] = auc_pr
            
            result['fprValues'] = [_format_values(fpr)]
            result['tprValues'] = [_format_values(tpr)]
            result['rocThresholds'] = [_format_values(roc_thresholds)]
            result['precisionValues'] = [_format_values(precisionValues)]
            result['recallValues'] = [_format_values(recallValues)]

    else:
        _logger.warn(
            "No positive example for " + str(index) + " and " +
            str(content_type))
    
    if len(result.keys()) == 0:
        result['ROC'] = 0
        result['fprValues'] = [np.zeros(2)]
        result['tprValues'] = [np.zeros(2)]
        result['rocThresholds'] = [np.zeros(2)]
        result['PR'] = 0
        result['precisionValues'] = [np.zeros(2)]
        result['recallValues'] = [np.zeros(2)]
        
    result = pd.DataFrame(result)
    if isinstance(index, str):
        result.index = [index]
    else:
        result.index = index
    
    return result


def _format_values(values):
    result = ','.join(values)
    return result


############################################################
# Downsampling curve (for better performance)
############################################################
# random downsampling
def _downsample_curve(x, y):
    if len(x) != len(y):
        raise Exception("x and y have different length: %d != %d" %
                        (len(x), len(y)))
    else:
        length = len(x)
    
    # logger.debug("Number of points before sampling: %d" % len(x))
    sample_size = min(length, config.EVALUATION_MAX_POINTS_ON_CURVE)
    np.random.seed(1)
    idx = np.random.choice(np.arange(length - 1), sample_size - 1, replace=False)
    
    # always keep the last element because it is special
    # (see sklearn documentation of pr_curve and roc_curve)
    idx = np.append(idx, length - 1)
    idx = np.sort(idx)
    result_x = x[idx]
    result_y = y[idx]
    # logger.debug("Number of points after sampling: %d" % len(result_x))
    
    return result_x, result_y


# interpolated downsampling to x values ranging from 0.01 to 1.00
def _downsample_curve2(x, y, thresholds):
    result_x = np.arange(0.01, 1.005, 0.01)
    
    f = scipy.interpolate.interp1d(x, y)
    result_y = f(result_x)
    
    f = scipy.interpolate.interp1d(x, thresholds)
    result_thresholds = f(result_x)
        
    return result_x, result_y, result_thresholds


def _convert_curve_to_str(x, y, thresholds):
    result_x = ["%.2f" % value for value in x]
    result_y = ["%f" % value for value in y]
    result_thresholds = ["%f" % value for value in thresholds]
    
    return result_x, result_y, result_thresholds
    

############################################################
# Evaluation
############################################################
def goadrich_pr_auc_score(y_true, y_score):
    _, _, auc_pr = _goadrich_precision_recall_curve(y_true, y_score)
    return auc_pr


def _goadrich_precision_recall_curve(y_true, probas_pred):
    if not ((len(y_true) > 0) and (sum(y_true) > 0)):
        return None, None, None
    
    precision, recall, _ = precision_recall_curve(y_true, probas_pred)
    precision, recall = _downsample_curve(precision, recall)
    
    prefix = 'wdvd_' + str(os.path.basename(config.OUTPUT_PREFIX))
    
    temp_directory = tempfile.TemporaryDirectory(prefix=prefix)
    
    with open(temp_directory.name + '/pr.csv', 'w', newline='') as csvfile:
        pr_writer = csv.writer(csvfile, delimiter='\t')
        for i in range(0, len(recall)):
            pr_writer.writerow([recall[i], precision[i]])
            
    posCount = sum(y_true)
    negCount = len(y_true) - posCount
    
    directoryOfThisFile = os.path.dirname(os.path.abspath(__file__))
    
    pathToJar = os.path.join(directoryOfThisFile, "../lib/auc.jar")
    pathToCSV = os.path.join(temp_directory.name, "pr.csv")
         
    try:
        output = subprocess.check_output(["java",
                                          "-jar",
                                          pathToJar,
                                          pathToCSV,
                                          "PR",
                                          str(int(posCount)),
                                          str(int(negCount))],
                                         universal_newlines=True)
    except subprocess.CalledProcessError as e:
        _logger.error(e.cmd)
        _logger.error(e.output)
        raise

    # logger.debug(output)
    
    with open (temp_directory.name + '/pr.csv.spr', 'r', newline='') as csvfile:
        pr_reader = csv.reader(csvfile, delimiter='\t')
        
        interpolated_precision = []
        interpolated_recall = []
        for row in pr_reader:
            interpolated_recall.append(row[0])
            interpolated_precision.append(row[1])
            
    match = re.search(
        '(?<=Area Under the Curve for Precision - Recall is )(.*)', output)
    auc_pr = float(match.group(0))
    
    temp_directory.cleanup()
            
    return interpolated_precision, interpolated_recall, auc_pr


# returns fpr and tpr values
def _roc_curve(y_true, y_score):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
    fpr, tpr, thresholds = _downsample_curve2(fpr, tpr, thresholds)
    fpr, tpr, thresholds = _convert_curve_to_str(fpr, tpr, thresholds)
    
    return fpr, tpr, thresholds


###############################################################################
# Plotting
###############################################################################
    
def _draw_ROC_plot(y_true, y_score, label):
    _logger.debug("Computing ROC Curve...")
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_score)
    fpr, tpr = _downsample_curve(fpr, tpr)
               
    _logger.debug("Plotting ROC Curve...")
    fig_name = "AUC-ROC"
    plt.figure(fig_name)
    plt.plot(fpr, tpr, label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('ROC Curve')
    plt.legend(loc="lower right", prop={'size': 6})
    plt.draw()
    plt.savefig(config.OUTPUT_PREFIX + "_" + fig_name + '.pdf')
    _logger.debug('ROC curve saved.')


def _draw_AUC_PR_plot(y_true, y_score, label):
    _logger.debug("Computing PR Curve...")
    interpolated_precision, interpolated_recall, auc_pr = \
        _goadrich_precision_recall_curve(y_true, y_score)
    _logger.debug("Plotting PR Curve....")
    fig_name = "AUC-PR"
    plt.figure(fig_name)
    plt.plot(interpolated_recall, interpolated_precision, label=label)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Interpolated Precision-Recall Curve')
    plt.legend(loc="upper right", prop={'size': 6}, bbox_to_anchor=(1.1, 1.1))
    plt.draw()
    _logger.info("Goadrich PR-AUC score: %s", auc_pr)
    plt.savefig(config.OUTPUT_PREFIX + "_" + fig_name + '.pdf')
    _logger.debug('PR curve saved.')
 
    
#############################################################
# Feature importances
#############################################################
def print_most_important_feature(importances, feature_names):
    index = np.argsort(importances)[::-1][0]
    _logger.info("Most important feature: %20s (%f)" %
                 (feature_names[index], importances[index]))

  
def print_feature_importances(importances, feature_names):
    # sort the indices of the features by decreasing importance
    # (the slicing operator [::-1] inverses the order)
    indices = np.argsort(importances)[::-1]
    
    # print the feature importances
    for f in range(len(indices)):
        _logger.info("%2d. %s (%f)" %
                     (f + 1, feature_names[indices[f]], importances[indices[f]]))
        
###############################################################################
# Misclassified examples
###############################################################################
        
# def print_feature_values(index, feature_names):
#     feature_values = data[index]
#     for i in range(len(feature_names)):
#         logger.info(feature_names[i] + ': ' + str(feature_values[i]))

      
# Prints some ids from the beginning, the middle and the end
def _print_subset_of_ids(indices, ids):
    if(len(indices) > 7):
        # At the beginning
        for revisionId in ids[indices[1:7]]:
            _logger.info("%d", revisionId)
            
        # logger.info(print_feature_values(indices[1]))
        _logger.info("----------------------------------------------------")
        # In the middle
        for revisionId in ids[indices[indices.size - 7:indices.size]]:
            _logger.info("%d", revisionId)
        _logger.info("----------------------------------------------------")
        # At the end
        for revisionId in ids[indices[np.random.randint(1, indices.size, size=7)]]:
            _logger.info("%d", revisionId)

      
def _print_some_misclassified_examples(y_true, y_pred, ids):
    false_negative_indices = np.where((y_true == True) & (y_pred == False))[0]
    _logger.info("Some false negatives of " + str(len(false_negative_indices)) + ': ')
    _print_subset_of_ids(false_negative_indices, ids)
    
#     logger.info("revision ID " + str(ids[false_negative_indices[0]]))
#     print_feature_values(false_negative_indices[0])
    
    false_positive_indices = np.where((y_true == False) & (y_pred == True))[0]
    _logger.info("Some false positives  " + str(len(false_positive_indices)) + ': ')
    _print_subset_of_ids(false_positive_indices, ids)
 
    
def _format_revisions(revisions):
    result = ""
    for i in range(len(revisions)):
        result = (
            result +
            "{:d} \t {:.2f}\n"
            .format(revisions['id'].iloc[i], revisions['y_score'].iloc[i]))
    return result

       
def _print_most_severely_misclassified_examples(y_true, y_score, ids):
    tmp = pd.DataFrame()
    tmp['id'] = ids
    tmp['y_true'] = y_true
    tmp['y_score'] = y_score
    
    positives = tmp[tmp['y_true'] == True]
    negatives = tmp[tmp['y_true'] == False]
    
    positives = positives.sort(['y_score'], ascending=True)[:5]
    negatives = negatives.sort(['y_score'], ascending=False)[:5]
    
    _logger.info('\nPositives scored very negatively:\n' +
                 _format_revisions(positives))
    _logger.info('\nNegatives scored very positively:\n' +
                 _format_revisions(negatives))

   
def _print_random_misclassified_examples(y_true, y_pred, y_score, ids):
    tmp = pd.DataFrame()
    tmp['id'] = ids
    tmp['y_true'] = y_true
    tmp['y_pred'] = y_pred
    tmp['y_score'] = y_score
    
    false_positives = tmp[(tmp['y_true'] == False) & (tmp['y_pred'] == True)]
    false_negatives = tmp[(tmp['y_true'] == True) & (tmp['y_pred'] == False)]
    
    number_fp_samples = min(10, len(false_positives))
    number_fn_samples = min(10, len(false_negatives))
                            
    if number_fp_samples > 0:
        false_positives = \
            false_positives.sample(number_fp_samples, random_state=1)
                            
    if number_fn_samples > 0:
        false_negatives = \
            false_negatives.sample(number_fn_samples, random_state=1)
    
    _logger.info("\nRandom false positives:\n" +
                 _format_revisions(false_positives))
    _logger.info("\nRandom false negatives:\n" +
                 _format_revisions(false_negatives))


def _print_random_misclassified_examples_by_content_type(
        y_true, y_pred, y_score, ids, contentTypes, contentType):
    contentType_select = (contentTypes.values == contentType)
    
    y_true = y_true[contentType_select]
    y_pred = y_pred[contentType_select]
    y_score = y_score[contentType_select]
    ids = ids[contentType_select]
    
    _logger.info("Misclassified " + contentType + " ...")
    _print_random_misclassified_examples(y_true, y_pred, y_score, ids)


# Saves examples where rollbackReverted is True and prob is less than 0.25
def save_misclassified_examples(prob, validation):
    _logger.debug("Saving misclassified examples (rollbackReverted=True, prob < 0.25)...")

    validation_X = validation.getX()
    
    misclassified_examples = pd.DataFrame(validation_X)
    misclassified_examples.columns = validation.getFeatureNames()
    misclassified_examples.insert(0, 'revisionId',
                                  validation.getRevisionIDs().reset_index(drop=True))
    misclassified_examples['rollbackReverted'] = validation.getY()
    
    mask = np.array(misclassified_examples['rollbackReverted'] > 0.5)
    misclassified_examples = misclassified_examples[mask]
    prob = prob[mask]
    
    misclassified_examples = misclassified_examples[np.array(prob < 0.25)]
    
    misclassified_examples.describe().to_csv(config.OUTPUT_PREFIX +
                                             "_misclassified_summary.csv")
    n = min(1000, len(misclassified_examples))
    misclassified_examples.sample(n).to_csv(config.OUTPUT_PREFIX +
                                            "_misclassified.csv")
    
    _logger.debug("Examples saved!")
