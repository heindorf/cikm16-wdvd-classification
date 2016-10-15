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

import datetime
import logging
import os
import platform
import sys

import matplotlib
import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn

import config
from . import constants

from .dataset import DataSet

from .pipeline import featureranking
from .pipeline import classification
from .pipeline import onlinelearning
from src.pipeline import optimization
from . import featurelist
from .pipeline import loading
from .pipeline import preprocessing
from .pipeline import statistics
from .queuelogger import QueueLogger

###############################################################################
# Remark: To execute this script the 64 bit version of Python, NumPy,
#         and SciPy is required (otherwise a MemoryError occurs)
###############################################################################
_logger = logging.getLogger()


# must be called by every process (and not only by the main process)
def set_output_prefix(output_prefix):
    config.OUTPUT_PREFIX = output_prefix
    
    create_directory(output_prefix)


def main(input_file):
    _print_system_info()
    _init_pandas()
    
    run_all(input_file)
    
    QueueLogger.stop()


def run_all(input_file):
    wdvd_features = featurelist.get_feature_list()
    filter_features = featurelist.get_filter_feature_list()
    ores_features = featurelist.get_ores_feature_list()
    
    if config.STATISTICS_ENABLED:
        data = loading.load_df(input_file, None)  # compute statistics for all columns
        statistics.compute_statistics(data)
    
    execute_pipeline(        input_file, 'VALIDATION', 'WDVD',   wdvd_features,   use_test_set=False, rank_features=config.FEATURE_RANKING_ENABLED, classify_groups=False                              , optimize=config.OPTIMIZATION_ENABLED)
    if config.BASELINES_ENABLED:
        execute_pipeline(    input_file, 'VALIDATION', 'FILTER', filter_features, use_test_set=False, rank_features=False                         , classify_groups=False                              , optimize=config.OPTIMIZATION_ENABLED)
        execute_pipeline(    input_file, 'VALIDATION', 'ORES',   ores_features,   use_test_set=False, rank_features=False                         , classify_groups=False                              , optimize=config.OPTIMIZATION_ENABLED)
    
    if config.USE_TEST_SET:
        execute_pipeline(    input_file, 'TEST',       'WDVD',   wdvd_features,   use_test_set=True, rank_features=config.FEATURE_RANKING_ENABLED, classify_groups=config.CLASSIFICATION_GROUPS_ENABLED, optimize=False)
        if config.BASELINES_ENABLED:
            execute_pipeline(input_file, 'TEST',       'FILTER', filter_features, use_test_set=True, rank_features=False                         , classify_groups=False                               , optimize=False)
            execute_pipeline(input_file, 'TEST',       'ORES',   ores_features,   use_test_set=True, rank_features=False                         , classify_groups=False                               , optimize=False)
    
    if config.ONLINE_LEARNING_ENABLED:
        onlinelearning.learn_online(input_file, wdvd_features, filter_features, ores_features)


def get_splitting_indices(data, use_test_set):
    training_set_start = constants.TRAINING_SET_START
    
    if use_test_set:
        validation_set_start = constants.TEST_SET_START
        test_set_start = constants.TAIL_SET_START
    else:
        validation_set_start = constants.VALIDATION_SET_START,
        test_set_start = constants.TEST_SET_START
    
    # transform revision id to index in data set
    training_set_start = DataSet.getIndexForRevisionIdFromDf(
        data, training_set_start)
    validation_set_start = DataSet.getIndexForRevisionIdFromDf(
        data, validation_set_start)
    test_set_start = DataSet.getIndexForRevisionIdFromDf(
        data, test_set_start)
    
    return training_set_start, validation_set_start, test_set_start
    

# one independent run (the data is loaded again)
def execute_pipeline(input_file, time_label, system_name, feature_list,
                     use_test_set, rank_features, classify_groups, optimize):
    QueueLogger.setContext(time_label, system_name)
    
    if rank_features | optimize | config.CLASSIFICATION_ENABLED:
        data = loading.load_df(input_file, featurelist.get_columns(feature_list))
    
        training_set_start, validation_set_start, test_set_start = \
            get_splitting_indices(data, use_test_set)
                                    
        # Starting the fitting at 0 yields the best results
        fit_slice = slice(0, validation_set_start)
          
        data = preprocessing.fit_transform(
            time_label, system_name, data, feature_list, fit_slice)
    
        # Splitting the data set into training and validation sets
        training = data[training_set_start:validation_set_start]
        validation = data[validation_set_start:test_set_start]
        _logger.debug("Training size: " + str(len(training)))
        _logger.debug("Validation size: " + str(len(validation)))

    if rank_features:
        featureranking.rank_features(training, validation)
        
    if optimize:
        optimization.optimize(training, validation)
    
    if config.CLASSIFICATION_ENABLED:
        classification.classify(training, validation, classify_groups)
 

def _print_system_info():
    
    if config.USE_TEST_SET:
        _logger.info("##################################################")
        _logger.info("# COMPUTATION ON TEST SET!!!")
        _logger.info("##################################################")
    
    # Host
    _logger.info("Host: " +  platform.node())
    _logger.info("Processor: " + platform.processor())
    _logger.info("Memory (in MB): " +
                 str(int(psutil.virtual_memory().total / 1024 / 1024)))
    
    # Operating system
    _logger.info("Platform: " + platform.platform())
    
    # Python
    _logger.info("Python interpreter: " + sys.executable)
    _logger.info("Python version: " + sys.version)
    
    # Libraries
    _logger.info("Numpy version: " + np.__version__)
    _logger.info("Scipy version: " + scipy.__version__)
    _logger.info("Pandas version: " + pd.__version__)
    _logger.info("Scikit-learn version: " + sklearn.__version__)
    _logger.info("Matplotlib version: " + matplotlib.__version__)
    _logger.info("Psutil version: " + psutil.__version__)
    
    # Classification
    _logger.info("Script file: " + os.path.abspath(sys.argv[0]))
    _logger.info("Script version: " + config.__version__)
    _logger.info("Script run time: " + str(datetime.datetime.now()))
    
    # Configuration
    for key, value in config._get_globals().items():
        if not key.startswith("__") and not key.startswith("_"):
            _logger.info(key + "=" + str(value))
    
    # plt.style.use('ggplot')

  
def create_directory(path_prefix):
    directory = os.path.split(config.OUTPUT_PREFIX)[0]
    
    if not os.path.exists(directory):
        os.mkdir(directory)
    elif os.listdir(directory) != []:
        input_var = input("Directory not empty: \"" +
                          directory +
                          "\". Remove all its content? (yes/no)")
        input_var = input_var.lower()
        if input_var in ['y', 'yes']:
            for the_file in os.listdir(directory):
                file_path = os.path.join(directory, the_file)
                os.unlink(file_path)


def _init_pandas():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 2000)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', 200)
