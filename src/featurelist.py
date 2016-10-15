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

import numpy as np

from src.ores import ores_featurelist
from .feature import Feature
from .transformers import BooleanImputer, MedianImputer, MinusOneImputer
from .transformers import CumUniqueTransformer, FrequencyTransformer, \
    LogTransformer, TimeTransformer, MeanTransformer, \
    TimeSinceLastRevisionTransformer


###############################################################################
# Features
###############################################################################
CURRENT_FEATURES = None


def get_feature_list():
    CURRENT_FEATURES = FEATURES

    result = _get_feature_list_from_dict(CURRENT_FEATURES)
    
    return result


def get_ores_feature_list():
    result = _get_feature_list_from_dict(ores_featurelist.BASELINE_FEATURES)
    return result


def get_filter_feature_list():
    FILTER_FEATURES = OrderedDict()
    FILTER_FEATURES['filter'] = OrderedDict()
    
    FILTER_FEATURES['filter']['filter'] = [
        Feature('revisionTag', [FrequencyTransformer()], 'revisionTagFreq')
    ]
    
    result = _get_feature_list_from_dict(FILTER_FEATURES)
    return result


def get_misc_feature_list():
    result = _get_feature_list_from_dict(MISC_FEATURES)
    return result


def get_meta_list():
    meta_list = [
        Feature('revisionId'),   # for determining training, validation and test set
        Feature('groupId'),      # for multiple instance learning
        Feature('contentType'),  # for evaluation by content type
        Feature('timestamp'),    # for determining datasets for online learning
    ]
    return meta_list


def get_label_list():
    label_list = [Feature('rollbackReverted')]
    return label_list


def get_columns(feature_list):
    meta_list = get_meta_list()
    feature_list = feature_list
    label_list = get_label_list()
    
    feature_list = meta_list + feature_list + label_list
    
    result = []
    
    for f in feature_list:
        inputNames = f.getInputNames()
        for inputName in inputNames:
            if (inputName != None) and (inputName not in result):
                result.append(inputName)
    
    return result


# based on the output names of the features
def get_feature_names(feature_list):
    result = []
    
    for f in feature_list:
        outputName = f.getOutputName()
        if (outputName != None) and (outputName not in result):
            result.append(outputName)
                
    return result


def _get_feature_list_from_dict(feature_dict):
    result = []
    for groupName, group in feature_dict.items():
        for subgroupName, subgroup in group.items():
            for feature in subgroup:
                feature.setGroup(groupName)
                feature.setSubgroup(subgroupName)
                result += [feature]
            
    return result

FEATURES = OrderedDict()
FEATURES['contextual'] = OrderedDict()
FEATURES['content'] = OrderedDict()

FEATURES['content']['character'] = [
    Feature('alphanumericRatio', [MedianImputer()]),
    Feature('asciiRatio', [MedianImputer()]),
    Feature('bracketRatio', [MedianImputer()]),
    Feature('digitRatio', [MedianImputer()]),
    Feature('latinRatio', [MedianImputer()]),
    Feature('longestCharacterSequence', [MinusOneImputer()]),
    Feature('lowerCaseRatio', [MedianImputer()]),
    Feature('nonLatinRatio', [MedianImputer()]),
    Feature('punctuationRatio', [MedianImputer()]),
    Feature('upperCaseRatio', [MedianImputer()]),
    Feature('whitespaceRatio', [MedianImputer()]),
]

FEATURES['content']['word'] = [
    Feature('badWordRatio', [MedianImputer()]),
    Feature('containsLanguageWord', []),
    Feature('containsURL'),
    Feature('languageWordRatio', [MedianImputer()]),
    Feature('longestWord', [MinusOneImputer()]),
    Feature('lowerCaseWordRatio', [MedianImputer()]),
    Feature('proportionOfQidAdded'),
    Feature('proportionOfLinksAdded'),

    Feature('upperCaseWordRatio', [MedianImputer()]),
]

FEATURES['content']['sentence'] = [
    Feature('commentCommentSimilarity', [MinusOneImputer()]),
    Feature('commentLabelSimilarity', [MinusOneImputer()]),
    Feature('commentSitelinkSimilarity', [MinusOneImputer()], 'commentSitelinkSimilarity'),
    Feature('commentTailLength', [MinusOneImputer()]),
]

FEATURES['content']['statement'] = [
    Feature('literalValue', [FrequencyTransformer()], 'literalValueFreq'),
    Feature('itemValue', [FrequencyTransformer()], 'itemValueFreq'),
    Feature('property', [FrequencyTransformer()], 'propertyFreq'),
]

FEATURES['contextual']['user'] = [
    Feature('isRegisteredUser', []),
    Feature('isPrivilegedUser', []),

    Feature('userCity', [FrequencyTransformer()], 'userCityFreq'),
    Feature('userCountry', [FrequencyTransformer()], 'userCountryFreq'),
    Feature('userCounty', [FrequencyTransformer()], 'userCountyFreq'),
    Feature('userContinent', [FrequencyTransformer()], 'userContinentFreq'),
    Feature('userName', [FrequencyTransformer()], 'userFreq'),
    Feature(['userName', 'itemId'], [CumUniqueTransformer()], 'cumUserUniqueItems'),
    Feature('userRegion', [FrequencyTransformer()], 'userRegionFreq'),
    Feature('userTimeZone', [FrequencyTransformer()], 'userTimeZoneFreq'),
]

FEATURES['contextual']['item'] = [
    Feature(['itemId', 'userName'], [CumUniqueTransformer(), LogTransformer()], 'logCumItemUniqueUsers'),
    Feature('itemId', [FrequencyTransformer(), LogTransformer()], 'logItemFreq'),
]

FEATURES['contextual']['revision'] = [
    Feature('commentLength', [MinusOneImputer()]),
    Feature('isLatinLanguage', [BooleanImputer()]),
    Feature('positionWithinSession'),
    Feature('revisionAction', [FrequencyTransformer()], 'revisionActionFreq'),
    Feature('revisionLanguage', [FrequencyTransformer()], 'revisionLanguageFreq'),
    Feature('revisionPrevAction', [FrequencyTransformer()], 'revisionPrevActionFreq'),
    Feature('revisionSubaction', [FrequencyTransformer()], 'revisionSubactionFreq'),
    Feature('revisionTag', [FrequencyTransformer()] , 'revisionTagFreq'),
]


###############################################################################
# Miscallaneous features (the features that did not make it into our final model)
###############################################################################
MISC_FEATURES = OrderedDict()
MISC_FEATURES['contextual'] = OrderedDict()
MISC_FEATURES['content'] = OrderedDict()

MISC_FEATURES
MISC_FEATURES['content']['character'] = [
    Feature('arabicRatio', [MedianImputer()]),
    Feature('bengaliRatio', [MedianImputer()]),
    Feature('brahmiRatio', [MedianImputer()]),
    Feature('cyrillicRatio', [MedianImputer()]),
    Feature('hanRatio', [MedianImputer()]),
    Feature('malayalamRatio', [MedianImputer()]),
    Feature('tamilRatio', [MedianImputer()]),
    Feature('teluguRatio', [MedianImputer()]),
]

MISC_FEATURES['content']['word'] = [
    Feature('proportionOfLanguageAdded'),
]

MISC_FEATURES['content']['sentence'] = [
    Feature('commentTail', [FrequencyTransformer()]),
]

MISC_FEATURES['content']['statement'] = [
    Feature('superItemId', [FrequencyTransformer()], 'superItemFreq'),
    Feature(['property', 'itemId'], [FrequencyTransformer()], 'propertyItemFreq'),
    Feature(['property', 'superItemId'], [FrequencyTransformer()], 'propertySuperItemFreq'),
    Feature(['itemValue', 'property'], [FrequencyTransformer()], 'itemValuePropertyFreq'),
    Feature(['literalValue', 'property'], [FrequencyTransformer()], 'literalValuePropertyFreq'),
]

MISC_FEATURES['contextual']['user'] = [
    Feature(['userName', 'timestamp'], [TimeSinceLastRevisionTransformer(), MedianImputer()], 'userTimeSinceLastRevision'),
    Feature(['userName', 'bytesIncrease'], [MeanTransformer()], 'userMeanBytesIncrease'),
    Feature(['userName', 'contentType'], [FrequencyTransformer()], 'userContentTypeFreq'),
]

MISC_FEATURES['contextual']['item'] = [
    Feature('hasListLabel', [BooleanImputer()]),
    Feature('isHuman', [BooleanImputer()]),
    Feature('labelCapitalizedWordRatio', [MedianImputer()]),
    Feature('labelContainsFemaleFirstName', [BooleanImputer()]),
    Feature('labelContainsMaleFirstName', [BooleanImputer()]),
                            
    Feature('numberOfLabels'),
    Feature('numberOfDescriptions'),
    Feature('numberOfAliases'),
    Feature('numberOfStatements'),
    Feature('numberOfSitelinks'),
    Feature('numberOfQualifiers'),
    Feature('numberOfReferences'),
    Feature('numberOfBadges'),
    
    Feature('numberOfLabels', [LogTransformer()], 'logNumberOfLabels'),
    Feature('numberOfDescriptions', [LogTransformer()], 'logNumberOfDescriptions'),
    Feature('numberOfAliases', [LogTransformer()], 'logNumberOfAliases'),
    Feature('numberOfStatements', [LogTransformer()], 'logNumberOfStatements'),
    Feature('numberOfSitelinks', [LogTransformer()], 'logNumberOfSitelinks'),
    Feature('numberOfQualifiers', [LogTransformer()], 'logNumberOfQualifiers'),
    Feature('numberOfReferences', [LogTransformer()], 'logNumberOfReferences'),
    Feature('numberOfBadges', [LogTransformer()], 'logNumberOfBadges'),
                          
]

MISC_FEATURES['contextual']['revision'] = [
    Feature('revisionSize'),
    Feature('revisionSize', [LogTransformer()], 'logRevisionSize'),
          
    Feature('timestamp', [TimeTransformer('hourOfDay')], 'revisionHourOfDay'),
    Feature('timestamp', [TimeTransformer('dayOfWeek')], 'revisionDayOfWeek'),
    Feature('timestamp', [TimeTransformer('dayOfMonth')], 'revisionDayOfMonth'),
]


def get_data_types():
    result = {
      
        # Meta features
        'revisionId': np.int32,
        'commentTail': str,
        'contentType': 'category',
        'groupId': np.int32,
        'itemId': np.int32,
        'timestamp': 'datetime',
        
         
        'userId': np.int32,
        'englishItemLabel': str,
        'superItemId': np.float32,
        'minorRevision': np.bool,
         
        # Character features
        'alphanumericRatio': np.float32,
        'asciiRatio': np.float32,
        'bracketRatio': np.float32,
        'digitRatio': np.float32,
        'latinRatio': np.float32,
        'longestCharacterSequence': np.float32,
        'lowerCaseRatio': np.float32,
        'nonLatinRatio': np.float32,
        'punctuationRatio': np.float32,
        'upperCaseRatio': np.float32,
        'whitespaceRatio': np.float32,
    
        # Misc character features
        'arabicRatio': np.float32,
        'bengaliRatio': np.float32,
        'brahmiRatio': np.float32,
        'cyrillicRatio': np.float32,
        'hanRatio': np.float32,
        'hindiRatio': np.float32,
        'malayalamRatio': np.float32,
        'tamilRatio': np.float32,
        'teluguRatio': np.float32,
         
        # Word features
        'badWordRatio': np.float32,
        'containsLanguageWord': np.bool,
        'containsURL': np.bool,
        'languageWordRatio': np.float32,
        'longestWord': np.float32,
        'lowerCaseWordRatio': np.float32,
        'proportionOfQidAdded': np.float32,
        'proportionOfLinksAdded': np.float32,
        'proportionOfLanguageAdded': np.float32,
        'upperCaseWordRatio': np.float32,
         
        # Misc word features
        'containsBadWord': np.bool,
        'containsLanguageWord2': np.bool,
         
        # Sentence features
        'commentCommentSimilarity': np.float32,
        'commentLabelSimilarity': np.float32,
        'commentSitelinkSimilarity': np.float32,
        'commentTailLength': np.float32,
         
        # Misc sentence features
        'wordsFromCommentInText': np.float32,
        'wordsFromCommentInTextWithoutStopWords': np.float32,
    
        # Statement features
        'property': 'category',
        'itemValue': 'category',
        'literalValue': str,
         
        # User features
        'isRegisteredUser': np.bool,
        'isPrivilegedUser': np.bool,
        'isBotUser': np.bool,
         
        'userCity': 'category',
        'userCountry': 'category',
        'userCounty': 'category',
        'userContinent': 'category',
        'userName': 'category',
        'userRegion': 'category',
        'userTimeZone': 'category',
         
        # Item features
        'hasListLabel': np.bool,
        'isHuman': np.bool,
        'labelCapitalizedWordRatio': np.float32,
        'labelContainsFemaleFirstName': np.bool,
        'labelContainsMaleFirstName': np.bool,
         
        # Misc item features
        'numberOfLabels': np.int32,
        'numberOfDescriptions': np.int32,
        'numberOfAliases': np.int32,
        'numberOfStatements': np.int32,
        'numberOfSitelinks': np.int32,
        'numberOfQualifiers': np.int32,
        'numberOfReferences': np.int32,
        'numberOfBadges': np.int32,
         
        'latestInstanceOfItemId': np.float32,
        'isLivingPerson': np.bool,
        'latestEnglishItemLabel': str,
         
        # Revision features
        'commentLength': np.float32,
        'isLatinLanguage': np.bool,
        'positionWithinSession': np.int32,
        'revisionAction': 'category',
        'revisionLanguage': 'category',
        'revisionPrevAction': 'category',
        'revisionSubaction': 'category',
        'revisionTag': 'category',
        'parentRevisionInCorpus': np.bool,
         
        # Misc revision features
        'param1': np.float32,
        'param3': str,
        'param4': str,
        'bytesIncrease': np.float32,
        'revisionSize': np.int32,
        'timeSinceLastRevision': np.float32,
         
        # Diff features
        'numberOfAliasesAdded': np.int32,
        'numberOfAliasesRemoved': np.int32,
         
        'numberOfBadgesAdded': np.int32,
        'numberOfBadgesRemoved': np.int32,
         
        'numberOfClaimsAdded': np.int32,
        'numberOfClaimsChanged': np.int32,
        'numberOfClaimsRemoved': np.int32,
        'numberOfDescriptionsAdded': np.int32,
        'numberOfDescriptionsChanged': np.int32,
        'numberOfDescriptionsRemoved': np.int32,
        'numberOfLabelsAdded': np.int32,
        'numberOfLabelsChanged': np.int32,
        'numberOfLabelsRemoved': np.int32,
        'numberOfSitelinksAdded': np.int32,
        'numberOfSitelinksChanged': np.int32,
        'numberOfSitelinksRemoved': np.int32,
        'numberOfIdentifiersChanged': np.int32,
        'numberOfSourcesAdded': np.int32,
        'numberOfSourcesRemoved': np.int32,
        'numberOfQualifiersAdded': np.int32,
        'numberOfQualifiersRemoved': np.int32,
         
        'englishLabelTouched': np.bool,
        'hasP21Changed': np.bool,
        'hasP27Changed': np.bool,
        'hasP54Changed': np.bool,
        'hasP569Changed': np.bool,
        'hasP18Changed': np.bool,
        'hasP109Changed': np.bool,
        'hasP373Changed': np.bool,
        'hasP856Changed': np.bool,
         
        # Labels
        'undoRestoreReverted': np.bool,
        'rollbackReverted': np.bool,
    }
    return result
