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

import math

import numpy as np
import pandas as pd


class DataSet:
    def __init__(self):
        self.__time_label = 'No time label'
        self.__system_name = 'No system'  # system name (e.g., WDVD, FILTER or ORES)
        self.__meta = pd.DataFrame()      # meta data about each instance (rows in X)
        self.__features = []              # list of feature names (columns in X)
        self.__X = None                   # numpy array of features
        self.__y = None                   # numpy array of labels
        
    def shallow_copy(self):
        result = DataSet()
        result.setTimeLabel(self.__time_label)
        result.setSystemName(self.__system_name)
        result.setMeta(self.__meta)
        result.setFeatures(self.__features)
        result.setX(self.__X)
        result.setY(self.__y)
        
        return result
    
    def __len__(self):
        return len(self.__y)
    
    def __getitem__(self, sliced):
        result = DataSet()
        result.setTimeLabel(self.__time_label)
        result.setSystemName(self.__system_name)
        result.setMeta(self.__meta[sliced])
        result.setFeatures(self.__features)
        result.setX(self.__X[sliced, :])
        result.setY(self.__y[sliced])
        
        return result
    
    def getX(self):
        return self.__X
    
    def setX(self, X):
        self.__X = np.ascontiguousarray(X)
    
    def getY(self):
        return self.__y
        
    def setY(self, y):
        self.__y = np.ascontiguousarray(y)
        
    def setFeatures(self, features):
        self.__features = features
        
    def getFeatures(self):
        return list(self.__features)
    
    def getGroups(self):
        result = [feature[0] for feature in self.__features]
        result = self.remove_duplicates(result)
        return result
    
    def getSubgroups(self):
        result = [(feature[0], feature[1]) for feature in self.__features]
        result = self.remove_duplicates(result)
        return result
    
    def getGroupNames(self):
        result = [feature[0] for feature in self.__features]
        result = self.remove_duplicates(result)
        return result
    
    def getSubgroupNames(self):
        result = [feature[1] for feature in self.__features]
        result = self.remove_duplicates(result)
        return result
    
    def getFeatureNames(self):
        result = [feature[2] for feature in self.__features]
        return result
    
    def setFeatureNames(self, feature_names):
        features = [('NO_GROUP', 'NO_SUBGROUP', feature_name)
                    for feature_name in feature_names]
        self.__features = features
        
    def getRevisionIDs(self):
        return self.__meta['revisionId']
    
    def getGroupIDs(self):
        return self.__meta['groupId']
    
    def getContentTypes(self):
        return self.__meta['contentType']
    
    def getUserName(self):
        return self.__meta['userName']
        
    def getMeta(self):
        return self.__meta
    
    def setMeta(self, meta):
        self.__meta = meta
        self.__meta.reset_index(drop=True, inplace=True)
        
    def setGroupIDs(self, groupIDs):
        self.__meta['groupId'] = groupIDs
        self.__meta.reset_index(drop=True, inplace=True)
        
    def getSystemName(self):
        return self.__system_name
    
    def setSystemName(self, system_name):
        self.__system_name = system_name
        
    def getTimeLabel(self):
        return self.__time_label
    
    def setTimeLabel(self, timeLabel):
        self.__time_label = timeLabel
        
    def select_features(self, feature_indices):
        result = self.shallow_copy()  # uses a lot of memory?!?
        result.setX(self.__X[:, feature_indices])
        
        features = [self.__features[feature_index]
                    for feature_index in feature_indices]
        result.setFeatures(features)
        return result
    
    def select_features_by_name(self, feature_names):
        cur_feature_names = self.getFeatureNames()
        feature_indices = [cur_feature_names.index(feature_name)
                           for feature_name in feature_names]
        
        result = self.select_features(feature_indices)
        return result
        
    def select_feature(self, feature_name):
        feature_indices = self.getFeatureNames().index(feature_name)
        result = self.select_features([feature_indices])
        return result
    
    def select_group(self, group_name):
        feature_indices = []
        for i in range(len(self.__features)):
            if self.__features[i][0] == group_name:
                feature_indices += [i]
        
        result = self.select_features(feature_indices)
        return result
    
    def select_subgroup(self, subgroup_name):
        feature_indices = []
        for i in range(len(self.__features)):
            if self.__features[i][1] == subgroup_name:
                feature_indices += [i]
        
        result = self.select_features(feature_indices)
        return result
    
    def add_feature(self, X, name):
        X = np.reshape(X, (-1, 1))
        self.setX(np.hstack((self.__X, X)))
        self.__features = self.__features + [name]
    
    # Does not necessarily create a copy
    def sample(self, fraction, seed=1):
        result = DataSet()
        
        np.random.seed(seed)
        selection = np.random.choice(
            len(self.__X),
            size=math.floor(fraction * len(self.__X)),
            replace=False)
        
        result.setMeta(self.__meta.iloc[selection])
        result.setX(self.__X[selection])
        result.setY(self.__y[selection])
        result.setFeatures(self.__features)
        result.setSystemName(self.__system_name)
        
        return result
    
    def apply_mask(self, mask):
        result = DataSet()
        
        result.setMeta(self.__meta.loc[mask, :])
        result.setX(self.__X[mask])
        result.setY(self.__y[mask])
        result.setFeatures(self.__features)
        result.setSystemName(self.__system_name)
        
        return result
    
    def append(self, dataset):
        result = DataSet()
        
        result.setMeta(self.__meta.append(dataset.__meta, ignore_index=True))
        result.setX(np.concatenate((self.__X, dataset.__X)))
        result.setY(np.concatenate((self.__y, dataset.__y)))
        result.setFeatures(self.__features)
        result.setSystemName(self.__system_name)
        
        return result
    
    def filter_body(self):
        mask = np.array(((self.getContentTypes() == 'STATEMENT') |
                         (self.getContentTypes() == 'SITELINK')))
        
        result = self.apply_mask(mask)
        return result
    
    def filter_head(self):
        mask = np.array(((self.getContentTypes() == 'TEXT')))
        
        result = self.apply_mask(mask)
        return result
    
    def get_positives(self):
        mask = (self.__y == True)
        
        result = self.apply_mask(mask)
        return result
    
    def undersample(self, n_times):
        nonRevertedIndices = np.where(self.__y == False)[0]
        selection = np.random.choice(
            nonRevertedIndices,
            size=math.floor(n_times * sum(self.__y)),
            replace=False)
        mask = (self.__y == True)  # select all reverted revisions
        mask[selection] = True  # additionally select the sampled revisions
        
        result = self.apply_mask(mask)
        return result
    
    # oversamples
    def oversample(self, n_times):
        indices = np.arange(len(self.__y))
        revertedIndices = np.where(self.__y == True)[0]
        
        indices = np.append(indices, [revertedIndices] * (n_times - 1))
        indices = np.sort(indices)
        
        result = DataSet()
        result.setMeta(self.__meta.iloc[indices])
        result.setX(self.__X[indices])
        result.setY(self.__y[indices])
        result.setFeatures(self.__features)
        result.setSystemName(self.__system_name)
        
        return result

    def random_split(self, fraction):
        np.random.seed(1)
        length = len(self.__X)
        selection = np.random.choice(
            length,
            size=math.floor(fraction * length),
            replace=False)
        mask1 = np.zeros(length, dtype=bool)
        mask1[selection] = True
        mask2 = ~mask1
        
        result1 = self.apply_mask(mask1)
        result2 = self.apply_mask(mask2)
        
        return result1, result2
    
    def split(self, fraction):
        length = len(self.__X)
        split_index = math.floor(length * fraction)
        
        return self[:split_index], self[split_index:]
    
    def getNumberOfFeatures(self):
        return self.getX().shape[1]
    
    @staticmethod
    def remove_duplicates(values):
        output = []
        seen = set()
        for value in values:
            if value not in seen:
                output.append(value)
                seen.add(value)
        return output
    
    def getVandalismFraction(self):
        vandalism_fraction = self.getY().mean()
        return vandalism_fraction
    
    def getIndexForRevisionId(self, revisionId):
        result = self.getIndexForRevisionIdFromDf(self.getMeta(), revisionId)
        return result
    
    def getIndexForDate(self, date):
        result = self.getIndexForDateFromDf(self.getMeta(), date)
        return result
    
    @staticmethod
    def getIndexForRevisionIdFromDf(df, revisionId):
        """
        Returns the first index that does NOT come before the
        specified revisionId.
        """
        result = df['revisionId'].searchsorted(revisionId)[0]
        return result
    
    @staticmethod
    def getIndexForDateFromDf(df, timestamp):
        """
        Returns the first index that does NOT come before the specified date.
        """
        result = df['timestamp'].searchsorted(timestamp)[0]
        return result
