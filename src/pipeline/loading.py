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
import os

import numpy as np
import pandas as pd

import config

from .. import featurelist
from .. import utils

_logger = logging.getLogger()


# loads features from disk and returns a pandas data frame
def load_df(input_file, columns):
    _logger.info("Loading data...")
    utils.collect_garbage()
    
    data = _read_cached_data(input_file, columns)
    
    _logger.info("Loading data... done.")
    
    return data


def _read_cached_data(filepath, columns):
    data_types = featurelist.get_data_types()
    
    if config.LOADING_USE_MEMORY_CACHE:
        global _data_cache
        
        if not ('_data_cache' in globals()):
            _logger.debug("reading csv...")
            # read all columns
            data = _read_data(filepath, None, data_types)
            _logger.debug("reading csv... done.")
            
            _data_cache = data
        
        _logger.debug("Copying data in memory...")
        data = _data_cache.copy()
        
        if not columns is None:
            _logger.debug("Memory usage before: " +
                          str(data.memory_usage(index=True, deep=True).sum()))
            data = data[columns]
            _logger.debug("Memory usage after:  " +
                          str(data.memory_usage(index=True, deep=True).sum()))
        _logger.debug("Copying data in memory... done.")
    elif config.LOADING_USE_DISK_CACHE:
        cache_file = os.path.basename(filepath) + '.cached.p'
        if not os.path.exists(cache_file):
            _logger.debug("Reading csv...")
            # read all columns
            data = _read_data(filepath, None, data_types)
            _logger.debug("Reading csv... done.")
            
            _logger.debug('Pickling...')
            data.to_pickle(cache_file)
            _logger.debug('Pickling...done')
            
        _logger.debug('Unpickling...')
        data = pd.read_pickle(cache_file)
        _logger.debug('Unpickling...done.')

        data = data[columns]
    # do not use any cache
    else:
        data = _read_data(filepath, columns, data_types)
        
    return data


# Reads the given features from the given file (either csv or csv.bz2).
def _read_data(filepath, columns, data_types):
    _logger.debug("Reading data from file...")

    if columns != None:
        _check_completeness_of_data_types(columns, data_types)
    
    # Pandas does not support reading categorical columns from CSV file.
    # Hence, we first read them as string and convert them later.
    replaced_data_types = data_types.copy()
    replaced_data_types = _replace_by_str('category', replaced_data_types)
    # Pandas does not support reading Boolean columns that contain NA values.
    # Hence, we first read them as string and convert them later.
    replaced_data_types = _replace_by_str(np.bool, replaced_data_types)
    replaced_data_types = _replace_by_str('datetime', replaced_data_types)

    data = pd.read_csv(
        filepath,
        quotechar='"',
        low_memory=True,
        keep_default_na=False,
        na_values=['NA', 'NaN', 'null', u'\ufffd'],
        dtype=replaced_data_types, usecols=columns,
        engine='c',
        buffer_lines=512 * 1024)
    
    _check_completeness_of_data_types(data.columns, data_types)
    
    _logger.debug("Categorizing data...")
    cat_columns = _get_affected_columns(data.columns, 'category', data_types)
    for column in cat_columns:
        data.loc[:, column] = data.loc[:, column].astype('category')

    _logger.debug("Converting Boolean data...")
    bool_columns = _get_affected_columns(data.columns, np.bool, data_types)
    true_false = {'T': True, 'F': False}
    for column in bool_columns:
        data[column] = data[column].map(true_false)
        
    _logger.debug("Converting datetime...")
    datetime_columns = _get_affected_columns(data.columns, 'datetime', data_types)
    for column in datetime_columns:
        data.loc[:, column] = pd.to_datetime(data.loc[:, column],
                                             format='%Y-%m-%dT%H:%M:%SZ',
                                             utc=True)
    _logger.debug("Reading data from file... done.")
    
    # Sorts the data by revision id which simplifies
    # the implementation of some transformers later.
    _sort_data_by_revision_id(data)
    
    return data


# Checks whether a datatype has been specified for every feature.
def _check_completeness_of_data_types(columns, data_types):
    for column in columns:
        if not column in data_types.keys():
            _logger.warn("No data type specified for column " + column)


# Given a list of columns and a dictionary of data_types,
# finds all columns that have the datatype data_type.
# Returns all columns having this datatype
def _get_affected_columns(columns, data_type, data_types):
    affected_columns = []
    for column in columns:
        if data_types[column] == data_type:
            affected_columns = affected_columns + [column]

    return affected_columns


# Returns a new datatype dictionary in which all occurrences of
# data_type_to_replace have been replaced by string.
#
# For example, this function is used to replace the datatypes 'category' and
# 'np.bool' by 'str' before reading the csv file.
def _replace_by_str(data_type_to_replace, data_types):
    # replaces data_type_to_replace by str in data_types
    new_data_types = data_types.copy()
    for key, value in data_types.items():
        if value == data_type_to_replace:
            new_data_types[key] = str
            
    return new_data_types


# sorts the pandas data frame data by revision id
def _sort_data_by_revision_id(data):
    _logger.debug("Sorting data by revision id...")
    
    data.sort_values(by='revisionId', ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    _logger.debug("Sorting data by revision id... done.")
    
    
# Only used for debugging when reading CSV file fails
def _binary_search(filename, columns):
    data_types = featurelist.get_data_types()
    
    _logger.debug("Binary search...")
    
    # file_handle = open(filename,'r', buffering = 10 * 1024 * 1024)
    skiprows = 0
    nrows = 2**24
    
    while nrows >= 1:
        _logger.debug("skiprows: %s, nrows %s" % (str(skiprows), str(nrows)))
    
        error = False
        try:
            pd.read_csv(filename,
                        quotechar='"',
                        low_memory=True,
                        keep_default_na=False,
                        na_values=['NA', 'NaN', 'null', u'\ufffd'],
                        encoding='utf-8',
                        dtype=data_types,
                        usecols=columns,
                        true_values=['T'],
                        false_values=['F'],
                        engine='c',
                        buffer_lines=512 * 1024,
                        skiprows=range(1, int(skiprows)),
                        nrows=nrows)
        except ValueError as msg:
            error = True
            _logger.debug("ValueError: " + str(msg))
            
        if not error:
            skiprows = skiprows + nrows
            
        nrows = nrows / 2
