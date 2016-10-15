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

import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.externals.joblib import Parallel, delayed

import config
from .. import evaluation


_logger = logging.getLogger()


#########################################################################
# Feature Ranking
#########################################################################
def rank_features(training, validation):
    _logger.info("Ranking features...")
    
    metrics = _compute_metrics_for_single_features(training, validation)
    
    group_metrics = _compute_metrics_for_feature_groups(training, validation)
    metrics = pd.concat([metrics, group_metrics], axis=0)
 
    _output_sorted_by_group(
        validation.getTimeLabel(), validation.getSystemName(),
        metrics, validation.getGroupNames(), validation.getSubgroupNames())
    
    _logger.info("Ranking features... done.")


# returns pandas data frame containing the metrics for each single feature
def _compute_metrics_for_single_features(training, validation):
    arguments = []
    for feature in validation.getFeatures():
        # each feature name is a tuple itself and
        # here we take the last element of this tuple
        training2 = training.select_feature(feature[-1])
        validation2 = validation.select_feature(feature[-1])
        argument = (training2, validation2, feature, )
        arguments.append(argument)
    
    result_list = Parallel(n_jobs=config.FEATURE_RANKING_N_JOBS,
                           backend='multiprocessing')(
        delayed(_compute_feature_metrics_star)(x) for x in arguments)
    
    result = pd.concat(result_list, axis=0)
    
    return result


def _compute_metrics_for_feature_groups(training, validation):
    arguments = []
    for subgroup in validation.getSubgroups():
        # each feature name is a tuple itself and here we take the last
        # element of this tuple
        training2 = training.select_subgroup(subgroup[-1])
        validation2 = validation.select_subgroup(subgroup[-1])
        argument = (training2, validation2, subgroup + ('ALL', ), )
        arguments.append(argument)
        
    for group in validation.getGroups():
        training2 = training.select_group(group)
        validation2 = validation.select_group(group)
        argument = (training2, validation2, (group, 'ALL', 'ALL'),)
        arguments.append(argument)
    
    result_list = Parallel(n_jobs=config.FEATURE_RANKING_N_JOBS,
                           backend='multiprocessing')(
        delayed(_compute_feature_metrics_star)(x) for x in arguments)
    
    result = pd.concat(result_list, axis=0)
    return result


# This method is called by multiple processes
def _compute_feature_metrics_star(args):
    return _compute_feature_metrics(*args)

   
# This method is called by multiple processes
def _compute_feature_metrics(training, validation, label):
    _logger.debug("Computing metrics for %s..." % str(label))
    
    index = pd.MultiIndex.from_tuples(
        [label], names=['Group', 'Subgroup', 'Feature'])
    
    _logger.debug("Using random forest...")
    clf = ensemble.RandomForestClassifier(random_state=1, verbose=0, n_jobs=-1)
    
    evaluation.fit(clf, training, index)
    
    y_pred, y_score = evaluation.predict(clf, validation, index)
    validation_result = evaluation.compute_feature_metrics_for_all_content_types(
        index, validation.getContentTypes(), validation.getY(), y_score, y_pred)
    
    # computing the feature metrics on the training set is useful for
    # identifying overfitting
    training_y_pred, training_y_score = evaluation.predict(clf, training, index)
    training_result = evaluation.compute_feature_metrics_for_content_type(
        index, training.getContentTypes(), 'ALL',
        training.getY(), training_y_score, training_y_pred)
    training_result.columns = list(itertools.product(
        ['TRAINING'], training_result.columns.values))
    
    result = pd.concat([validation_result, training_result], axis=1)
    
    return result


def _output_sorted_by_auc_pr(time_label, system_name, metrics):
    """
    Outputs the metrics sorted by area under precision-recall curve
    """
    _logger.debug("output_sorted_by_auc_pr...")
    metrics.sort_values([('ALL', 'PR')], ascending=False, inplace=True)
    metrics.to_csv(config.OUTPUT_PREFIX + "_" + time_label + "_" +
                   system_name + "_feature_ranking.csv")

    latex = metrics.loc[:, evaluation.COLUMN_LIST]
    # latex.reset_index(drop=True, inplace=True)
    latex.to_latex(config.OUTPUT_PREFIX + "_" + time_label + "_" +
                   system_name + "_feature_ranking.tex", float_format='{:.3f}'.format)

    n_features = min(9, len(metrics) - 1)
    selection = metrics.iloc[0:n_features] \
                       .loc[:, [('ALL', 'Feature'), ('ALL', 'PR')]]
    _logger.info("Top 10 for all content\n" +
                 (selection.to_string(float_format='{:.4f}'.format)))
    _logger.debug("output_sorted_by_auc_pr... done.")

    
def _output_sorted_by_group(
        time_label, system_name, metrics, group_names, subgroup_names):
    """
    Outputs the metrics sorted by group and by PR-AUC within a group
    """
    _logger.debug('_output_sorted_by_group...')
    
    metrics['_Group'] = metrics.index.get_level_values('Group')
    metrics['_Subgroup'] = metrics.index.get_level_values('Subgroup')
    metrics['_Feature'] = metrics.index.get_level_values('Feature')
    
    group_names = group_names
    subgroup_names = ['ALL'] + subgroup_names
   
    # Define the order of groups and subgroups
    metrics['_Group'] = metrics['_Group'].astype('category').cat.set_categories(
        group_names, ordered=True)
    metrics['_Subgroup'] = metrics['_Subgroup'].astype('category').cat.set_categories(
        subgroup_names, ordered=True)

    # Sort the features by AUC_PR and make sure the subgroup is always shown
    # before the single features
    metrics['_Order'] = metrics[('ALL', 'PR')]
    # without this line, the following line causes a PerformanceWarning
    metrics.sort_index(inplace=True)
    metrics.loc[(metrics['_Feature'] == 'ALL'), '_Order'] = 1.0
    
    metrics.sort_values(by=['_Group', '_Subgroup', '_Order'],
                        ascending=[True, True, False], inplace=True)

    metrics.to_csv(config.OUTPUT_PREFIX + "_" + time_label + "_" +
                   system_name + "_feature_groups.csv")
    
    formatted_names = metrics.apply(_compute_feature_name, axis=1)
    metrics.set_index(formatted_names, inplace=True)
    evaluation.print_metrics_to_latex(
        metrics, config.OUTPUT_PREFIX + "_" + time_label + "_" +
        system_name + "_feature_groups.tex")
    
    _logger.debug('_output_sorted_by_group... done.')

    
def _compute_feature_name(row):
    group = row['_Group'].iloc[0]
    subgroup = row['_Subgroup'].iloc[0]
    feature = row['_Feature'].iloc[0]
    
    # Is group?
    if subgroup == 'ALL' and feature == 'ALL':
        result = "\\quad %s" % group
    # Is subgroup?
    elif feature == 'ALL':
        result = "\\quad\quad %s" % subgroup
    # Is feature?
    else:
        result = "\\quad\\quad\\quad %s" % feature
        
    return result
    

# For example, plot_name can be 'features' or 'groups'
def _plot_top_pr_curves(time_label, system_name, metrics, plot_name):
    for content in metrics.columns.levels[0]:
        value = metrics.loc[:, content].copy()
        if 'PR' in value:
            value = value.sort_values('PR', ascending=False)
            # value.reset_index(drop=True, inplace=True)
            if len(value) > 0:
                _plot_auc_pr(value, content, '_%s_%s_plot_top_%s_%s.pdf' %
                             (time_label, system_name, plot_name, content))


def _plot_auc_pr(table, title, filename):
    num_plots = min(10, len(table))
    feature_names = pd.DataFrame(table.index.get_level_values('Feature'))
    plt.figure(title)
    colormap = plt.get_cmap('gist_ncar')
    colors = [colormap(i) for i in np.linspace(0, 0.9, num_plots)]
    # Matplotlib 1.4.3, deprecated in Matplotlib 1.5.1
    # plt.gca().set_color_cycle(colors)
    plt.gca().set_prop_cycle(plt.cycler('color', colors))  # Matplotlib 1.5.1
    for i in range(num_plots):
        recallValues = table.iloc[i].loc['recallValues'].split(',')
        precisionValues = table.iloc[i].loc['precisionValues'].split(',')
        featureName = feature_names.iloc[i].loc['Feature']
        
        if len(recallValues) > 0 and len(precisionValues) > 0:
            plt.plot(recallValues, precisionValues,
                     label=featureName, linewidth=2.0)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(title)
            plt.legend(loc="upper right",
                       prop={'size': 10},
                       bbox_to_anchor=(1.1, 1.1))
            plt.draw()
     
            plt.savefig(config.OUTPUT_PREFIX + filename)
    plt.ion()
    plt.show()
    plt.ioff()
