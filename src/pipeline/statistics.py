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

from collections import OrderedDict
import logging

import numpy as np
import pandas as pd

import config

from .. import constants
from ..dataset import DataSet

_logger = logging.getLogger()

# Use thousand separator and no decimal points
_FLOAT_FORMAT = '{:,.0f}'.format


# Input: dataframe data
def compute_statistics(data):
    _logger.info("Computing statistics...")
 
    _compute_feature_statistics(data)
    
    _compute_corpus_statistics(data)
    _compute_dataset_statistics(data)
    _compute_session_statistics(data)
    
    # computes some statistics about selected features
    _compute_special_feature_statistics(data)
    
    _logger.info("Computing statistics... done.")

    
def _compute_feature_statistics(data):
    _logger.debug("Computing descriptive statistics...")
    
    data.describe(include='all').to_csv(
        config.OUTPUT_PREFIX + "_feature_statistics.csv")
    
    _logger.debug("Computing descriptive statistics... done.")
    
    
def _compute_corpus_statistics(data):
    """
    The Wikidata Vandalism Corpus WDVC-2015 in
    terms of total unique users, items, sessions, and revisions with a
    breakdown by content type and by vandalism status (vandalism/non-vandalism).
    """
    
    def compute_data_frame(data):
        headMask = data['contentType'] == 'TEXT'
        stmtMask = (data['contentType'] == 'STATEMENT')
        sitelinkMask = (data['contentType'] == 'SITELINK')
        bodyMask = (stmtMask | sitelinkMask)
        
        result = OrderedDict()
        result['Entire corpus'] = compute_column_group(data)
        result['Item head'] = compute_column_group(data[headMask])
        result['Item body'] = compute_column_group(data[bodyMask])
        # result['STATEMENT'] = compute_column_group(data[stmtMask])
        # result['SITELINK'] = compute_column_group(data[sitelinkMask])
        
        result = pd.concat(result, axis=1, keys=result.keys())
        return result
    
    def compute_column_group(data):
        vandalismMask = (data['rollbackReverted'] == True)
        regularMask = (data['rollbackReverted'] == False)
        
        result = OrderedDict()
        result['Total'] = compute_column(data)
        result['Vandalism'] = compute_column(data[vandalismMask])
        result['Regular'] = compute_column(data[regularMask])
        
        result = pd.concat(result, axis=1, keys=result.keys())
        return result
    
    def compute_column(data):
        result = pd.Series()
        
        result['Revisions'] = data['revisionId'].nunique()
        result['Sessions'] = data['groupId'].nunique()
        result['Items'] = data['itemId'].nunique()
        result['Users'] = data['userName'].nunique()
        
        return result
    
    statistics = compute_data_frame(data)
    statistics.to_csv(config.OUTPUT_PREFIX + "_corpus_statistics.csv")
    
    statistics = _round_to_thousands(statistics)
    statistics.to_latex(
        config.OUTPUT_PREFIX + "_corpus_statistics.tex",
        float_format=_FLOAT_FORMAT)
    
    _logger.info(statistics)
 
    
def _compute_dataset_statistics(data):
    """
    Evaluation datasets for training, validation, and test in
    terms time period covered, revisions, sessions, items, and users.
    """
    
    def compute_data_frame(data):
        _logger.debug("Splitting statistics...")
        training_set_start_index = DataSet.getIndexForRevisionIdFromDf(data, constants.TRAINING_SET_START)
        validation_set_start_index = DataSet.getIndexForRevisionIdFromDf(data, constants.VALIDATION_SET_START)
        test_set_start_index = DataSet.getIndexForRevisionIdFromDf(data, constants.TEST_SET_START)
        test_set_end_index = DataSet.getIndexForRevisionIdFromDf(data, constants.TAIL_SET_START)
        
        trainingSet = data[training_set_start_index:validation_set_start_index]
        validationSet = data[validation_set_start_index:test_set_start_index]
        testSet = data[test_set_start_index:test_set_end_index + 1]
        
        result = []
        result.append(compute_splitting_statistics_row(trainingSet, 'Training'))
        result.append(compute_splitting_statistics_row(validationSet, 'Validation'))
        result.append(compute_splitting_statistics_row(testSet, 'Test'))
        
        result = pd.concat(result, axis=0)
        return result
    
    def compute_splitting_statistics_row(data, label):
        result = pd.Series()
        result['Revisions'] = data['revisionId'].nunique()
        result['Sessions'] = data['groupId'].nunique()
        result['Items'] = data['itemId'].nunique()
        result['Users'] = data['userName'].nunique()
        
        result = result.to_frame().transpose()
        
        result.index = [label]
        
        return result
        
    result = compute_data_frame(data)
    
    # logger.info("Splitting statistics:\n" + str(result))
    result.to_csv(config.OUTPUT_PREFIX + "_dataset_statistics.csv")
    
    result = _round_to_thousands(result)
    result.to_latex(config.OUTPUT_PREFIX + "_dataset_statistics.tex",
                    float_format=_FLOAT_FORMAT)
    
    _logger.debug("Splitting statistics... done.")


def _compute_session_statistics(data):
    """
    Computes some statistics about revision groups (a.k.a. sessions)
    """
    groupsize = data.groupby(by='groupId').size()
    groupsize.name = 'groupsize'
    groupsize = groupsize.to_frame()
    groupsize['groupId'] = groupsize.index
    
    # merge revisions with groupsizes
    joinedGroupsize = data[['revisionId', 'groupId']].join(
        groupsize, on='groupId', how='left', lsuffix='_left', rsuffix='_right')
    
    # distribution of group sizes among revisions)
    counts = joinedGroupsize['groupsize'].value_counts()
    
#     logger.info("Distribution of group sizes of revisions:\n " + str(counts))
    
    # revisions which are not alone in their group
    partOfLargerGroup = sum(counts[counts.index > 1])
    allRevisions = len(data)
    _logger.info(
        "Fraction of revisions which are not alone in their session: " +
        "%d / %d = %.2f" %
        (partOfLargerGroup, allRevisions, partOfLargerGroup / allRevisions))


def _compute_special_feature_statistics(data):
    def compute_feature_statistics_main(data):
        compute_vandalism_probability(data['revisionTag'], data['rollbackReverted'], get_revision_tag_mapping(data))
        compute_vandalism_probability(data['languageWordRatio'], data['rollbackReverted'], get_language_word_ratio_mapping(data))
        compute_vandalism_probability(data['revisionLanguage'], data['rollbackReverted'], get_revision_language_mapping(data))
        compute_vandalism_probability(data['userCountry'], data['rollbackReverted'])
        
    def compute_vandalism_probability(data, y_true, mapping=None, name=None):
        TOP_K = 5
        
        if name is None:
            name = data.name
                   
        data = data.astype(str)

        _logger.debug("Computing vandalism probability for feature %s" % name)
            
        result = compute_vandalism_probability2(data, y_true)
        result.to_csv(config.OUTPUT_PREFIX + "_feature_%s.csv" % (name))
        
        if mapping != None:
            result = apply_mapping(result, mapping)
            result.to_csv(config.OUTPUT_PREFIX + "_feature_%s_mapped.csv" % (name))
                   
        nonNan = result.index.tolist()
        if 'nan' in nonNan:
            nonNan.remove('nan')
        top = nonNan[:TOP_K]
        truncated_mapping = {value: value if value in top else 'misc' for value in nonNan}
        truncated_mapping['nan'] = 'nan'
        
        truncated_result = apply_mapping(result, truncated_mapping)
        
        # insert first row of table which shows statistics for nonNan values
        nonNan_mapping = {value: 'nonNan' for value in nonNan}
        nonNan_mapping['nan'] = np.NaN  # the mapping for 'nan' should be undefined such that it does not appear in the result
        nonNan_result = apply_mapping(result, nonNan_mapping)
        truncated_result = nonNan_result.append(truncated_result)
        
        truncated_result.to_csv(config.OUTPUT_PREFIX + "_feature_%s_truncated.csv" % (name))
        
        truncated_result[['vandalismRevisions', 'totalRevisions']] = \
            _round_to_thousands(truncated_result[['vandalismRevisions', 'totalRevisions']])
        truncated_result['vandalismProbability'] = \
            truncated_result['vandalismProbability'] * 100
        truncated_result.to_latex(
            config.OUTPUT_PREFIX + "_feature_%s_truncated.tex" % (name),
            float_format=_FLOAT_FORMAT,
            formatters={'vandalismProbability': '{:,.2f}%'.format})
        
    def compute_vandalism_probability2(data, y_true):
        result = y_true.groupby(data).sum().to_frame()
        result.columns = ["vandalismRevisions"]
        result['totalRevisions'] = y_true.groupby(data).size()
        result['vandalismProbability'] = (result['vandalismRevisions'] /
                                          result['totalRevisions'])
        result.sort_values(by='vandalismRevisions', ascending=False, inplace=True)

        return result
    
    def apply_mapping(df, mapping):
        series = df.index.to_series()
        
        mapped = series.replace(mapping)
        
        # groupby ignores nan values (hence, nan must be encoded as string 'nan')
        result = df.groupby(mapped).sum()
        result['vandalismProbability'] = (result['vandalismRevisions'] /
                                          result['totalRevisions'])
        result.sort_values(by='vandalismRevisions', ascending=False, inplace=True)
        
        # move 'misc' and 'nan' values to the end
        result = move_row_to_end(result, 'misc')
        result = move_row_to_end(result, 'nan')

        return result
    
    def update_vandalism_probability(df):
        df['vandalismProbability'] = (df['vandalismRevisions'] /
                                      df['totalRevisions'])
    
    def move_row_to_end(df, idx_value):
        result = df.copy()
        
        idx = result.index
        if idx_value in idx:
            loc = idx.get_loc(idx_value)
            idx = idx.delete(loc)
            idx = idx.insert(idx.size, idx_value)
            result = result.reindex(idx)
        
        return result
    
    def get_revision_language_mapping(data):
        language_values = data['revisionLanguage'].astype('str').unique()
        project_suffixes = ['wiki', 'wikisource', 'wikiquote', 'wikinews', 'wikivoyage']
        language_variant_prefixes = ['de-', 'en-']  # Further information: https://www.wikidata.org/wiki/Wikidata:Requests_for_comment/Labels_and_descriptions_in_language_variants
        
        language_dict = {}
        for language_value in language_values:
            key = language_value
            value = language_value
            
            # make empty string count as revision without language
            if value == '':
                value = 'nan'

            # make enwiki, enwikisource, enwikiquote, ... all count as English
            for suffix in project_suffixes:
                if language_value.endswith(suffix):
                    value = language_value[:-len(suffix)]
                    
            # make en, en, ca, en-gb, ...all count as English
            # make de, de-at, de-ch, ... all count as German
            for language_variant in language_variant_prefixes:
                if value.startswith(language_variant):
                    value = value[:len(language_variant) - 1]
                    
            language_dict[key] = value
                
        return language_dict
    
    # Further information:
    # Tags: https://www.wikidata.org/wiki/Special:Tags
    # Abuse Filter: https://www.wikidata.org/wiki/Special:AbuseFilter
    def get_revision_tag_mapping(data):
        revision_tag_values = data['revisionTag'].astype('str').unique()
        editing_tools = ['OAuth CID', 'HHVM']
        revision_tag_dict = {}
        
        for revision_tag_value in revision_tag_values:
            if any(tool_str in revision_tag_value for tool_str in editing_tools):
                revision_tag_dict[revision_tag_value] = 'EDITING TOOLS'
            elif revision_tag_value == '':
                revision_tag_dict[revision_tag_value] = 'nan'
            else:
                revision_tag_dict[revision_tag_value] = 'ABUSE FILTER'
                
        return revision_tag_dict
    
    def get_language_word_ratio_mapping(data):
        word_ratio_values = data['languageWordRatio'].astype('str').unique()
        word_ratio_dict = {}
        
        for value in word_ratio_values:
            if float(value) > 0:
                word_ratio_dict[value] = True
            elif (float(value) == -1.0) or value == 'nan':
                word_ratio_dict[value] = 'nan'
            else:
                word_ratio_dict[value] = False
                
        return word_ratio_dict
    
    compute_feature_statistics_main(data)

    
def _round_to_thousands(statistics):
    statistics = statistics / 1000  # numbers in thousand
    statistics = statistics.round()
    
    return statistics
