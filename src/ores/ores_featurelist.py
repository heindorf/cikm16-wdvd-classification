from collections import OrderedDict

from src.feature import Feature
from src.transformers import BooleanImputer


# source: https://github.com/wiki-ai/wb-vandalism/blob/31d74f8a50a8c43dd446d41cafee89ada5a051f8/wb_vandalism/feature_lists/wikidata.py
# features as of October 28, 2015 (version of the Birthday present)
BASELINE_FEATURES = OrderedDict()
BASELINE_FEATURES['baseline'] = OrderedDict()

BASELINE_FEATURES['baseline']['diff'] = [
    Feature('numberOfSitelinksAdded'),
    Feature('numberOfSitelinksRemoved'),
    Feature('numberOfSitelinksChanged'),
    
    Feature('numberOfLabelsAdded'),
    Feature('numberOfLabelsRemoved'),
    Feature('numberOfLabelsChanged'),
    
    Feature('numberOfDescriptionsAdded'),
    Feature('numberOfDescriptionsRemoved'),
    Feature('numberOfDescriptionsChanged'),
    
    Feature('numberOfAliasesAdded'),
    Feature('numberOfAliasesRemoved'),
    
    Feature('numberOfClaimsAdded'),
    Feature('numberOfClaimsRemoved'),
    Feature('numberOfClaimsChanged'),
    
    Feature('numberOfIdentifiersChanged'),
    Feature('englishLabelTouched'),
    
    Feature('numberOfSourcesAdded'),
    Feature('numberOfSourcesRemoved'),
            
    Feature('numberOfQualifiersAdded'),
    Feature('numberOfQualifiersRemoved'),

    Feature('numberOfBadgesAdded'),
    Feature('numberOfBadgesRemoved'),
    
    Feature('proportionOfQidAdded'),
    Feature('proportionOfLinksAdded'),
    Feature('proportionOfLanguageAdded'),
    
    Feature('hasP21Changed'),   # sex or gender
    Feature('hasP27Changed'),   # country of citizenship
    Feature('hasP54Changed'),   # member of sports team
    Feature('hasP569Changed'),  # date of birth
    Feature('hasP18Changed'),   # image
    Feature('hasP109Changed'),  # signature
    Feature('hasP373Changed'),  # commons category
    Feature('hasP856Changed'),  # official website
    
    Feature('numberOfLabels'),
    Feature('numberOfDescriptions'),
    Feature('numberOfAliases'),
    Feature('numberOfStatements'),
    Feature('numberOfSitelinks'),
    Feature('numberOfQualifiers'),
    Feature('numberOfReferences'),
    Feature('numberOfBadges'),
    
    Feature('isLivingPerson'),
    Feature('isHuman', [BooleanImputer()]),
    Feature('isBotUser'),
    Feature('isRegisteredUser'),
]
