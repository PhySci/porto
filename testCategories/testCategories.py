"""
Purpose of the script is to identify true categorical features and ordinary ones.
Final score will be calculated is assumption that one (or a few) features is improve model's performance if they are
categorical
"""

import numpy as np
import pandas as pd

from catboost import Pool, CatBoostClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import logging

import os

import itertools


def trainCatBoost(trainSet, targetSet, params={'depth': 7, 'rate': 0.055, 'l2': 8, 'T': 1.5}, folds=5, maxIter=2000,
                  verbose=False, cat_features=[], logger=[]):
    """
    Split dataset on 5 folds, train and validate model, estimate score.
    :param trainSet: pandas Dataframe
    :param targetSet: target values
    :param params: dictionary of model's parameters
    :param folds:  amount of folds
    :param maxIter: amount of iteration (early stop will be applied)
    :param verbose:  verbose output?
    :param cat_features: list of categorical features (int)
    :return: AUR-ROC score
    """

    logger.info("Features %s", cat_features)
    # create log directory
    dirName = '/tmp/porto/catboost/testFeatures/'+'-'.join(map(str, cat_features))

    try:
        os.makedirs(dirName)
    except Exception as inst:
        print inst  # __str__ allows args to be printed directly

    treeList = list()
    scoreList = list()
    prob = np.zeros([trainSet.shape[0]])

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    for i, (train_index, val_index) in enumerate(skf.split(trainSet, targetSet)):
        model = CatBoostClassifier(verbose=verbose, iterations=maxIter, thread_count=4, eval_metric="AUC",
                                   depth=params['depth'], learning_rate=params['rate'],
                                   l2_leaf_reg=params['l2'], bagging_temperature=params['T'],
                                   od_type='Iter', od_wait=20,
                                   train_dir=dirName +"/"+ str(i), random_seed=i)

        # create pool
        trainPool = Pool(trainSet.iloc[train_index], targetSet.iloc[train_index], cat_features = cat_features,
                         feature_names=trainSet.columns.tolist())

        valPool = Pool(trainSet.iloc[val_index], targetSet.iloc[val_index], cat_features = cat_features,
                         feature_names=trainSet.columns.tolist())

        # fit and estimate the model
        model.fit(trainPool, eval_set=valPool, use_best_model=True)
        prob[val_index] = model.predict_proba(valPool)[:, 1]
        localScore = roc_auc_score(targetSet.iloc[val_index], prob[val_index])

        treeList.append(model.tree_count_)
        scoreList.append(localScore)

        logger.info('Fold: %d, tree amount: %d, score: %f', i, model.tree_count_, localScore)

    score = roc_auc_score(targetSet, prob)
    #print cat_features, ': ', score
    return score


def getData():
    """
    Return Pandas dataframe for training.
    :return: pandas dataframe
    """

    rawData = pd.read_csv('../data/train.csv', index_col='id')

    # drop _calc_ features
    dropList = list()
    for fName in rawData.columns.tolist():
        if fName.find('_calc_') > (-1):
            dropList.append(fName)
    df = rawData.drop(dropList, axis=1)

    # squared feature "ps_car_15"
    df = df.assign(ps_car_15_mod = np.power(df.ps_car_15,2).astype(int)).drop("ps_car_15", axis = 1)

    # inverse one-hot-encoding for ind_06 % ind_09
    df = df.assign(ps_ind_69_cat = 0*df.ps_ind_06_bin+df.ps_ind_07_bin+2*df.ps_ind_08_bin+3*df.ps_ind_09_bin)
    df.drop(['ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin'], inplace=True, axis = 1)

    #drop "ind_14"
    df.drop('ps_ind_14', axis = 1, inplace = True)

    return [df.drop("target", axis= 1), df.target]


def get_cat_features(df):
    """
    Return list of categorical features
    :param df: Pandas dataframe
    :return: dictionary of categorical features {featureId: featureName}
    """

    catFeatures = {}

    for i, fName in enumerate(df.columns.tolist()):
        if fName == 'id' or fName == 'target':
            continue
        if df.loc[:, fName].dtypes != 'int':
            continue
        if len(df.loc[:, fName].unique()) < 20:
            catFeatures.update({i: fName})

    return catFeatures


def main():

    # set logging info
    logging.basicConfig(filename='testCatFeatures.log', format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    [X, y] = getData()
    catFeatures = get_cat_features(X)

    resDict = dict()

    # loop over features
    for featureIds in itertools.combinations(catFeatures.keys(), 2):
        score = trainCatBoost(X, y, cat_features= list(featureIds), verbose= False, logger= logger)
        resDict.update({featureIds: score})
        print 'Feature: ',featureIds, '. Score is', score
        logger.info("Iteration: %s. Score: %f", featureIds, score)

    resDf = pd.DataFrame.from_dict(resDict)
    resDf.to_pickle('2features.pcl')

    resDict = dict()
    # loop over features
    for featureIds in itertools.combinations(catFeatures.keys(), 3):
        score = trainCatBoost(X, y, cat_features= list(featureIds), verbose= False, logger= logger)
        resDict.update({featureIds: score})
        print 'Feature: ',featureIds, '. Score is', score
        logger.info("Iteration: %s. Score: %f", featureIds, score)

    resDf = pd.DataFrame.from_dict(resDict)
    resDf.to_pickle('3features.pcl')


if __name__ == "__main__":
    main()