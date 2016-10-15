import logging

import sklearn.cross_validation
import sklearn.ensemble
from sklearn.metrics import auc, make_scorer, precision_recall_curve


logger = logging.getLogger()

SCORERS = {}


# Taken from https://github.com/wiki-ai/revscoring/blob/d95d32b24ff13b4e83839ea3281f661392f35868/revscoring/utilities/metrics.py
# on December 27, 2015
def pr_auc_score(y_true, y_score):
    """
    Generates the Area Under the Curve for precision and recall.
    """
    precision, recall, _ = \
        precision_recall_curve(y_true, y_score[:, 1])
    return auc(recall, precision, reorder=True)

pr_auc_scorer = make_scorer(pr_auc_score, greater_is_better=True,
                            needs_proba=True)
    
SCORERS['pr_auc'] = pr_auc_scorer


def cross_validation(clf, data, scorer, cv):
    logger.info("Cross validation clf:\n %s" % str(clf))
    logger.info("Cross validation cv: %s" % str(cv))
    logger.info("Cross validation scorer: %s" % str(scorer))
    
    scores = sklearn.cross_validation.cross_val_score(
        clf, data.getX(), data.getY(), scoring=scorer, cv=cv, n_jobs=-1)
    
    logger.info("Cross validation mean: %f, std: %f" %
                (scores.mean(), scores.std()))
    logger.info("Cross validation scores: %s" % str(scores))


# Based on
# https://github.com/wiki-ai/revscoring/blob/d95d32b24ff13b4e83839ea3281f661392f35868/revscoring/utilities/tune.py
# https://github.com/wiki-ai/wb-vandalism/blob/master/tuning_reports/wikidata.reverted.roc_auc.md
# https://github.com/wiki-ai/wb-vandalism/blob/master/tuning_reports/wikidata.reverted.pr_auc.md
# cross validation performed as Halfaker (using sklearn cross_val_score, 5 folds)
def cross_validation_Halfaker(training, validation):
    logger.debug("Cross validation...")
    data = training.append(validation)
    data = data.undersample(1)
    logger.debug("Data size: %d" % len(data))
    logger.debug("Vandalism: %d" % data.getY().sum())
    
    clf = sklearn.ensemble.RandomForestClassifier(
        verbose=0, n_jobs=-1, random_state=1)
    
    cv = sklearn.cross_validation.StratifiedKFold(
        data.getY(), n_folds=5, shuffle=True, random_state=None)
    cross_validation(clf, data, 'roc_auc', cv)
    cross_validation(clf, data, SCORERS['pr_auc'], cv)
    
    
def classify(training, validation):
    cross_validation_Halfaker(training, validation)
