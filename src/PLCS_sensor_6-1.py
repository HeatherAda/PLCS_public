"""
sensor, cluster EM
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import re
import itertools
import pylab as pl
import operator

from collections import defaultdict as dd
from collections import Counter as ct

from sklearn.cluster import KMeans
from sklearn.mixture import DPGMM

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TV
# from sklearn.cross_validation import StratifiedKFold
# from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.preprocessing import normalize

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from collections import OrderedDict

import dawid_skene as DSEM

from datetime import datetime
import os
import copy

from multiprocessing import Pool

import sklearn.utils.validation
from sklearn.exceptions import NotFittedError

import sys

sys.path.insert(0, "../utils")

from utils import *

from random import randint

random.seed(10)
np.random.seed(10)

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def get_name_features(names):
    name = []
    for i in names:
        s = re.findall('(?i)[a-z]{2,}', i)
        name.append(' '.join(s))

    cv = CV(analyzer='char_wb', ngram_range=(3, 4))
    fn = cv.fit_transform(name).toarray()

    return fn


class _Corpus:

    def __init__(self):
        self.m_category = None
        self.m_multipleClass = None
        self.m_feature = None
        self.m_label = None
        #self.m_transferLabel = None
        self.m_transferLabelList = []
        self.m_auditorLabel = None
        self.m_initialExList = []
        #
        self.m_sensorName = {}
        self.m_labelNum = None
        self.m_label2Idx = {}
        self.m_idx2Label = {}
        self.m_accTransferList = []

    def initCorpus(self, featureMatrix, labelArray, transferLabelArrayList, auditorLabelArray, initialExList, category,
                   multipleClass, distinct_label_list, sensorName = None):
        if sensorName is not None:
            self.m_sensorName = {id: sensorName[id] for id in range(len(sensorName))}
        self.m_category = category
        print("category", category)
        self.m_multipleClass = multipleClass
        print("multipleClass", multipleClass)
        self.m_feature = featureMatrix
        self.m_label = labelArray
        # self.m_transferLabel = transferLabelArray
        self.m_transferLabelList = transferLabelArrayList

        self.m_initialExList = initialExList
        self.m_auditorLabel = auditorLabelArray
        #
        self.m_labelNum = len(distinct_label_list)
        for i, label in enumerate(distinct_label_list):
            self.m_label2Idx[label] = i
            self.m_idx2Label[i] = label
        for i in range(len(transferLabelArrayList)):
            self.m_accTransferList.append(accuracy_score(labelArray, transferLabelArrayList[i]))
        print("transfer acc: "+str(self.m_accTransferList))


class _ActiveClf:
    def __init__(self, category, multipleClass, labelNum, StrongLabelNumThreshold):
        self.m_activeClf = None

        self.m_strongLabelNumThresh = StrongLabelNumThreshold

        self.m_initialExList = []

        self.m_labeledIDList = []

        self.m_unlabeledIDList = []

        self.m_accList = []

        self.m_gold = None
        self.m_weak = None  # dataset used to train weak oracles

        self.m_train = None

        self.m_test = None

        self.m_multipleClass = multipleClass

        self.m_category = category

        #
        self.m_accList_gold = []

        self.m_accTransfer_pred = []

        self.m_useEM = True
        self.m_useTransfer = True
        self.m_useClf = False

        self.m_labelNum = labelNum

        self.m_weakOracles = []
        self.m_confid_threshold = 0.0

        self.m_weakOracleCM_pred = None
        self.m_weakOracleCM_true = None
        self.m_prior_pred = None
        self.m_prior_true = None

        self.m_weakOracleAcc_pred = []  # estimated weak oracle accuracy
        self.m_weakOracleAcc_true = []  # true acc on untrn data
        self.m_rmsePredWeakAcc = []  # RMSE

        self.m_initParamWithGold = True
        self.m_algType = "PLCS"  # supervised, AL, PL, PLCS

        self.m_correctAnsNum = []  # (correctNum, ansNum) by weak
        self.m_revisedAnsNum = []  # (clfCorrectRevise, clfRevised, StrongRevise)

        self.m_id_select_conf_list = []
        # self.m_weakLabelPrecisionList = []
        # self.m_weakLabelRecallList = []
        # self.m_weakLabelAccList = []  # acc when answer

        self.m_weakOracleCM_cluster_pred = {} # {cluster:{oracle: CM}}
        self.m_weakOracleAcc_cluster_true = {}
        self.m_weakOracleAcc_cluster_pred = {} # {cluster:{oracle: acc}}

        self.m_accTransfer_cluster_pred = {}

        self.m_accList_al = []

        #
        self.tao = 0
        self.alpha_ = 1

        self.clf = LinearSVC()
        self.clf_al = LinearSVC()
        self.ex_id = dd(list)
        self.ex_id_cs = dd(list)

        self.id2cluster = {}

    def tool(self, a1, a2): # true, pred
        n00 = 0
        n01 = 0
        n10 = 0
        n11 = 0
        for i in range(len(a1)):
            if a1[i] == 1.0:
                if a2[i] == 1.0:
                    n11 += 1
                else:
                    n10 += 1
            else:
                if a2[i] == 1.0:
                    n01 += 1
                else:
                    n00 += 1
        return n00, n01, n10, n11

    def initActiveClf(self, initialSampleList, gold, weak, train, test):
        self.m_initialExList = initialSampleList
        self.m_gold = gold
        self.m_weak = weak
        self.m_train = train
        self.m_test = test

        # true prior
        label_train = corpusObj.m_label[train]
        labelNum_train = {}
        for lb in corpusObj.m_label2Idx:
            lb_num = list(label_train).count(lb)
            labelNum_train[lb] = lb_num
        for lb in labelNum_train:
            labelNum_train[lb] = float(labelNum_train[lb]) / len(train)
        self.m_prior_true = [labelNum_train[i] for i in sorted(labelNum_train.iterkeys())]

        # generate weak oracles and use gold task to estimate their quality
        if self.m_algType == "PLCS":
            # split the dataset into different parts
            trn_random = self.m_weak
            # random.shuffle(trn_random)
            orcl_trn_data = []

            cur_pos = 0
            weak_rate = []#[0.0125, 0.05, 0.1, 0.25, 0.5]  # [0.05, 0.1, 0.15, 0.25, 0.4] # number of weak oracles
            for i in range(len(weak_rate)):
                sample_num = int(len(trn_random) * weak_rate[i])
                # sampledTrainIDs = []
                if len(set(trn_random)) <= 1:
                    print("error in weak oracle training dataset")
                while True:
                    sampledTrainIDs = random.sample(trn_random, sample_num)
                    label_sample = corpusObj.m_label[sampledTrainIDs]
                    if len(set(label_sample)) > 1:
                        break
                trn_random = list(set(trn_random) - set(sampledTrainIDs))
                orcl_trn_data.append(sampledTrainIDs)

                w_feature = corpusObj.m_feature[sampledTrainIDs]
                w_label = corpusObj.m_label[sampledTrainIDs]
                if self.m_multipleClass:
                    w_ocl = LR(multi_class="multinomial", solver='lbfgs', random_state=3, fit_intercept=False)
                else:
                    w_ocl = LR(random_state=3)
                w_ocl.fit(w_feature, w_label)
                self.m_weakOracles.append(w_ocl)

            all_sampleID = [i for i in range(len(corpusObj.m_label))]

            # true acc
            for i in range(len(self.m_weakOracles)):
                # acc on untrained data
                untrn = [j for j in all_sampleID if j not in orcl_trn_data[i]]
                untrn_features = corpusObj.m_feature[untrn]
                untrn_labels = corpusObj.m_label[untrn]
                wocle_acc_untrn = self.m_weakOracles[i].score(untrn_features, untrn_labels)
                self.m_weakOracleAcc_true.append(wocle_acc_untrn)

            # true CM
            nObservers = len(self.m_weakOracles)
            partOracleStart = 0
            if self.m_useTransfer:
                nObservers += len(corpusObj.m_transferLabelList)
                partOracleStart += len(corpusObj.m_transferLabelList)

            if self.m_useEM:
                error_rates = np.zeros([nObservers, corpusObj.m_labelNum, corpusObj.m_labelNum])

                # CM of transfer
                if self.m_useTransfer:
                    untrn = all_sampleID
                    untrn_features = corpusObj.m_feature[untrn]
                    untrn_labels = corpusObj.m_label[untrn]

                    for i in range(len(corpusObj.m_transferLabelList)):
                        untrn_labels_pred = corpusObj.m_transferLabelList[i][untrn]
                        #n00, n01, n10, n11 = self.tool(untrn_labels, untrn_labels_pred)

                        for j in range(len(untrn)):
                            label_idx_pred = corpusObj.m_label2Idx[untrn_labels_pred[j]]
                            label_idx_true = corpusObj.m_label2Idx[untrn_labels[j]]
                            error_rates[i][label_idx_true][label_idx_pred] += 1

                        for j in range(corpusObj.m_labelNum):
                            sum_over_row = np.sum(error_rates[i, j, :])
                            if sum_over_row > 0:
                                error_rates[i, j, :] = error_rates[i, j, :] / float(sum_over_row)

                # CM of other oracles
                for i in range(len(self.m_weakOracles)):
                    if self.m_useTransfer:
                        oracle_id = i + partOracleStart
                    else:
                        oracle_id = i
                    # acc on untrained data
                    untrn = [j for j in all_sampleID if j not in orcl_trn_data[i]]
                    untrn_features = corpusObj.m_feature[untrn]
                    untrn_labels = corpusObj.m_label[untrn]
                    untrn_labels_pred = self.m_weakOracles[i].predict(untrn_features)
                    n00, n01, n10, n11 = self.tool(untrn_labels, untrn_labels_pred)

                    for j in range(len(untrn)):
                        label_idx_pred = corpusObj.m_label2Idx[untrn_labels_pred[j]]
                        label_idx_true = corpusObj.m_label2Idx[untrn_labels[j]]

                        error_rates[oracle_id][label_idx_true][label_idx_pred] += 1

                    for j in range(corpusObj.m_labelNum):
                        sum_over_row = np.sum(error_rates[oracle_id, j, :])
                        if sum_over_row > 0:
                            error_rates[oracle_id, j, :] = error_rates[oracle_id, j, :] / float(sum_over_row)

                self.m_weakOracleCM_true = error_rates

                if self.m_useTransfer:
                    acc_true_list = self.computeAccWithCM(self.m_weakOracleCM_true, self.m_prior_true)
                    self.m_weakOracleAcc_true = acc_true_list[partOracleStart:]
                    #self.m_accTransfer_pred = acc_true_list[:partOracleStart]  # will not be used, will be changed
                else:
                    self.m_weakOracleAcc_true = self.computeAccWithCM(self.m_weakOracleCM_true, self.m_prior_true)

            # gold task
            else:
                if self.m_initParamWithGold:
                    gold_features = corpusObj.m_feature[self.m_gold]
                    gold_labels = corpusObj.m_label[self.m_gold]
                    for i in range(len(self.m_weakOracles)):
                        wocle_acc_cur = self.m_weakOracles[i].score(gold_features, gold_labels)  # acc on gold
                        self.m_weakOracleAcc_pred.append(wocle_acc_cur)  # estimated accuracy of weak oracles

                    # calculate the RMSE for estimation of weak oracle accuracy
                    RMSE = 0.0
                    for i in range(len(self.m_weakOracleAcc_pred)):
                        RMSE += math.pow(self.m_weakOracleAcc_pred[i] - self.m_weakOracleAcc_true[i], 2)

                    if self.m_useTransfer:
                        for i in range(len(corpusObj.m_transferLabelList)):
                            self.m_accTransfer_pred.append(accuracy_score(gold_labels, corpusObj.m_transferLabelList[i][self.m_gold]))
                            RMSE += math.pow(corpusObj.m_accTransferList[i] - self.m_accTransfer_pred[i], 2)

                    RMSE = math.sqrt(RMSE / nObservers)
                    print("RMSE: "+str(RMSE))
                    self.m_rmsePredWeakAcc.append(RMSE)

        # classifier
        if self.m_multipleClass:
            self.m_activeClf = LR(multi_class="multinomial", solver='lbfgs', random_state=3, fit_intercept=False)
        else:
            self.m_activeClf = LR(random_state=3)

    def select_example(self, corpusObj):

        #selectedID = random.sample(self.m_unlabeledIDList, 1)[0]
        # print("selectedID", selectedID)
        #return selectedID

        unlabeledIdScoreMap = {}
        unlabeledIdNum = len(self.m_unlabeledIDList)

        for unlabeledIdIndex in range(unlabeledIdNum):
            # print("unlabeledIdIndex", unlabeledIdIndex)
            unlabeledId = self.m_unlabeledIDList[unlabeledIdIndex]

            score = self.getClassifierMargin(corpusObj, unlabeledId)

            unlabeledIdScoreMap[unlabeledId] = score

        if len(unlabeledIdScoreMap) == 0:
            print("zero unlabeledId")

        sortedUnlabeledIdList = sorted(unlabeledIdScoreMap, key=unlabeledIdScoreMap.__getitem__, reverse=True)

        selectedID = sortedUnlabeledIdList[0]
        score4SelectedID = unlabeledIdScoreMap[selectedID]

        return selectedID

    def select_example_ByConf(self, id_conf):
        minconfID = -1
        minconf = 1.1
        for id in id_conf:
            if id_conf[id] < minconf:
                minconf = id_conf[id]
                minconfID = id

        return minconfID

    def getClassifierMargin(self, corpusObj, unlabeledId):
        labelPredictProb = self.m_activeClf.predict_proba(corpusObj.m_feature[unlabeledId].reshape(1, -1))[0]

        labelProbMap = {}  ##labelIndex: labelProb
        labelNum = len(labelPredictProb)
        for labelIndex in range(labelNum):
            labelProbMap.setdefault(labelIndex, labelPredictProb[labelIndex])

        sortedLabelIndexList = sorted(labelProbMap, key=labelProbMap.__getitem__, reverse=True)

        maxLabelIndex = sortedLabelIndexList[0]
        subMaxLabelIndex = sortedLabelIndexList[1]

        maxLabelProb = labelProbMap[maxLabelIndex]
        subMaxLabelProb = labelProbMap[subMaxLabelIndex]

        margin = maxLabelProb - subMaxLabelProb

        margin = 0 - margin

        return margin

    '''
    select weak oracles
    strategy: 
        1. random
        2. top k
        3. all
    '''
    def selectWeakOracles(self, argv):
        selected_oracle_IDs = []
        if argv[0] == "random":
            oracleNum = 1  # random 1 by default
            if (len(argv)>1):
                oracleNum = argv[1]
            selected_oracle_IDs = random.sample(range(len(self.m_weakOracles)), oracleNum)
        else:
            wocl_acc = [i for i in enumerate(self.m_weakOracleAcc_pred)]
            wocl_acc.sort(key=lambda x:x[1], reverse=True)  # sort weak oracle IDs by their accuracy
            if argv[0] == "top":
                oracleNum = 1  # top 1 by default
                if (len(argv) > 1):
                    oracleNum = min(argv[1], len(self.m_weakOracles))
                selected_oracle_IDs = [i for i, acc in wocl_acc[:oracleNum]]
            if argv[0] == "all":
                selected_oracle_IDs = [i for i in range(len(self.m_weakOracles))]
        return selected_oracle_IDs

    '''
        visit weak oracles to gather answers
        aggregate answers
    '''
    def answerAggregation(self, idx, selected_oracle_IDs, argvs):
        feature = corpusObj.m_feature[idx].reshape(1, -1)
        gt_label = corpusObj.m_label[idx]

        answer_worcl_list = []
        # gather answers from selected oracles
        for i in selected_oracle_IDs:
            cur_woracle = self.m_weakOracles[i]
            cur_ans = cur_woracle.predict(feature)
            answer_worcl_list.append(cur_ans[0])

        # aggregate answers
        votes = [0.0] * self.m_labelNum  # transfer
        confidence = 0.0
        if argvs[0] == "MV":  # Majority voting
            for j in answer_worcl_list:
                ans_idx = corpusObj.m_label2Idx[j]
                votes[ans_idx] += 1
            # transfer
            if self.m_useTransfer:
                for i in range(len(corpusObj.m_transferLabelList)):
                    transferLabel = corpusObj.m_transferLabelList[i][idx]
                    ans_idx = corpusObj.m_label2Idx[transferLabel]
                    votes[ans_idx] += 1

            # classifier
            if self.m_useClf:
                clfLabel = self.m_activeClf.predict(feature)[0]
                ans_idx = corpusObj.m_label2Idx[clfLabel]
                votes[ans_idx] += 1

            confidence = max(votes)/sum(votes)
            ans_agg = corpusObj.m_idx2Label[votes.index(max(votes))]
        if argvs[0] == "WV":  # Weighted voting
            for j in range(len(answer_worcl_list)):
                cur_ans = answer_worcl_list[j]
                ans_idx = corpusObj.m_label2Idx[cur_ans]
                votes[ans_idx] += self.m_weakOracleAcc_pred[selected_oracle_IDs[j]]
            # transfer
            if self.m_useTransfer:
                for i in range(len(corpusObj.m_transferLabelList)):
                    transferLabel = corpusObj.m_transferLabelList[i][idx]
                    ans_idx = corpusObj.m_label2Idx[transferLabel]
                    votes[ans_idx] += self.m_accTransfer_pred[i]
            # classifier
            if self.m_useClf:
                clfLabel = self.m_activeClf.predict(feature)[0]
                acc_cur_clf = self.m_accList_gold[-1]
                ans_idx = corpusObj.m_label2Idx[clfLabel]
                votes[ans_idx] += acc_cur_clf

            confidence = max(votes) / sum(votes)
            ans_agg = corpusObj.m_idx2Label[votes.index(max(votes))]
        if argvs[0] == "BV":  # Bayes voting
            for r in range(self.m_labelNum):  # for each answer option
                p_mul_r = 1.0
                p_mul_notr = 1.0
                cur_label = corpusObj.m_idx2Label[r]
                for j in range(len(answer_worcl_list)):  # for each oracle
                    cur_ans = answer_worcl_list[j]
                    if cur_ans == cur_label:
                        p_mul_r *= self.m_weakOracleAcc_pred[selected_oracle_IDs[j]]
                    else:
                        p_mul_notr *= (1-self.m_weakOracleAcc_pred[selected_oracle_IDs[j]])/(self.m_labelNum - 1)
                # transfer
                if self.m_useTransfer:
                    for i in range(len(corpusObj.m_transferLabelList)):
                        transferLabel = corpusObj.m_transferLabelList[i][idx]
                        # ans_idx = corpusObj.m_label2Idx[transferLabel]
                        if transferLabel == cur_label:
                            p_mul_r *= self.m_accTransfer_pred[i]
                        else:
                            p_mul_notr *= (1 - self.m_accTransfer_pred[i])/(self.m_labelNum - 1)

                # classifier
                if self.m_useClf:
                    clfLabel = self.m_activeClf.predict(feature)[0]
                    acc_cur_clf = self.m_accList_gold[-1]
                    if clfLabel == cur_label:
                        p_mul_r *= acc_cur_clf
                    else:
                        p_mul_notr *= (1 - acc_cur_clf) / (self.m_labelNum - 1)

                votes[r] = p_mul_r * p_mul_notr
            confidence = max(votes) / sum(votes)
            ans_agg = corpusObj.m_idx2Label[votes.index(max(votes))]
        return ans_agg, confidence, ans_agg == gt_label

    '''
            visit weak oracles to gather answers by clusters
            aggregate answers
        '''

    def answerAggregation_cluster(self, idx, selected_oracle_IDs, argvs):
        feature = corpusObj.m_feature[idx].reshape(1, -1)
        gt_label = corpusObj.m_label[idx]
        cluster_id = self.id2cluster[idx]

        answer_worcl_list = []
        # gather answers from selected oracles
        for i in selected_oracle_IDs:
            cur_woracle = self.m_weakOracles[i]
            cur_ans = cur_woracle.predict(feature)
            answer_worcl_list.append(cur_ans[0])

        # aggregate answers
        votes = [0.0] * self.m_labelNum  # transfer
        confidence = 0.0
        if argvs[0] == "MV":  # Majority voting
            for j in answer_worcl_list:
                ans_idx = corpusObj.m_label2Idx[j]
                votes[ans_idx] += 1
            # transfer
            if self.m_useTransfer:
                for i in range(len(corpusObj.m_transferLabelList)):
                    transferLabel = corpusObj.m_transferLabelList[i][idx]
                    ans_idx = corpusObj.m_label2Idx[transferLabel]
                    votes[ans_idx] += 1

            # classifier
            if self.m_useClf:
                clfLabel = self.m_activeClf.predict(feature)[0]
                ans_idx = corpusObj.m_label2Idx[clfLabel]
                votes[ans_idx] += 1

            confidence = max(votes) / sum(votes)
            ans_agg = corpusObj.m_idx2Label[votes.index(max(votes))]
        if argvs[0] == "WV":  # Weighted voting
            for j in range(len(answer_worcl_list)):
                cur_ans = answer_worcl_list[j]
                ans_idx = corpusObj.m_label2Idx[cur_ans]
                #votes[ans_idx] += self.m_weakOracleAcc_pred[selected_oracle_IDs[j]]
                votes[ans_idx] += self.m_weakOracleAcc_cluster_pred[cluster_id][selected_oracle_IDs[j]]
            # transfer
            if self.m_useTransfer:
                for i in range(len(corpusObj.m_transferLabelList)):
                    transferLabel = corpusObj.m_transferLabelList[i][idx]
                    ans_idx = corpusObj.m_label2Idx[transferLabel]
                    #votes[ans_idx] += self.m_accTransfer_pred[i]
                    votes[ans_idx] += self.m_accTransfer_cluster_pred[cluster_id][i]
            # classifier
            if self.m_useClf:
                clfLabel = self.m_activeClf.predict(feature)[0]
                acc_cur_clf = self.m_accList_gold[-1]
                ans_idx = corpusObj.m_label2Idx[clfLabel]
                votes[ans_idx] += acc_cur_clf

            confidence = max(votes) / sum(votes)
            ans_agg = corpusObj.m_idx2Label[votes.index(max(votes))]
        if argvs[0] == "BV":  # Bayes voting
            for r in range(self.m_labelNum):  # for each answer option
                p_mul_r = 1.0
                p_mul_notr = 1.0
                cur_label = corpusObj.m_idx2Label[r]
                for j in range(len(answer_worcl_list)):  # for each oracle
                    cur_ans = answer_worcl_list[j]
                    if cur_ans == cur_label:
                        #p_mul_r *= self.m_weakOracleAcc_pred[selected_oracle_IDs[j]]
                        p_mul_r *= self.m_weakOracleAcc_cluster_pred[cluster_id][selected_oracle_IDs[j]]
                    else:
                        #p_mul_notr *= (1 - self.m_weakOracleAcc_pred[selected_oracle_IDs[j]]) / (self.m_labelNum - 1)
                        p_mul_notr *= (1 - self.m_weakOracleAcc_cluster_pred[cluster_id][selected_oracle_IDs[j]]) / (self.m_labelNum - 1)
                # transfer
                if self.m_useTransfer:
                    for i in range(len(corpusObj.m_transferLabelList)):
                        transferLabel = corpusObj.m_transferLabelList[i][idx]
                        # ans_idx = corpusObj.m_label2Idx[transferLabel]
                        if transferLabel == cur_label:
                            #p_mul_r *= self.m_accTransfer_pred[i]
                            niu = self.m_accTransfer_cluster_pred[cluster_id][i]
                            p_mul_r *= self.m_accTransfer_cluster_pred[cluster_id][i]
                        else:
                            #p_mul_notr *= (1 - self.m_accTransfer_pred[i]) / (self.m_labelNum - 1)
                            niu = self.m_accTransfer_cluster_pred[cluster_id][i]
                            p_mul_notr *= (1 - self.m_accTransfer_cluster_pred[cluster_id][i]) / (self.m_labelNum - 1)

                # classifier
                if self.m_useClf:
                    clfLabel = self.m_activeClf.predict(feature)[0]
                    acc_cur_clf = self.m_accList_gold[-1]
                    if clfLabel == cur_label:
                        p_mul_r *= acc_cur_clf
                    else:
                        p_mul_notr *= (1 - acc_cur_clf) / (self.m_labelNum - 1)

                votes[r] = p_mul_r * p_mul_notr
            confidence = max(votes) / sum(votes)
            ans_agg = corpusObj.m_idx2Label[votes.index(max(votes))]
        return ans_agg, confidence, ans_agg == gt_label

    def generateCleanDataByCrowd_cluster(self, taskIDList, confid_thresh=None, min_wt = 0.0):
        cleanIDTrain = []
        cleanLabelTrain = []

        total = 0
        correct_num = 0

        id_confid = {}

        if confid_thresh is None:
            confid_thresh = self.m_confid_threshold

        # filter clusters where all transferred oracles are not trustful
        rmIds = []
        if self.m_useTransfer:
            for cid in self.ex_id_cs:
                max_wt = max([self.m_accTransfer_cluster_pred[cid][i] for i in range(len(corpusObj.m_transferLabelList))])
                if max_wt < min_wt:
                    rmIds.extend(self.ex_id_cs[cid])
        rmIds = {id: 1 for id in rmIds}
        taskIDList = [id for id in taskIDList if id not in rmIds]

        # aggregation
        for idx in taskIDList:

            orc_argv = ["all"]
            selected_oracle_IDs = self.selectWeakOracles(orc_argv)
            # answer aggregation
            ans_argv = ["WV"]
            agg_ans, confidence_curr, correct = self.answerAggregation_cluster(idx, selected_oracle_IDs, ans_argv)

            id_confid[idx] = confidence_curr

            if confidence_curr >= confid_thresh:
                cleanIDTrain.append(idx)
                cleanLabelTrain.append(agg_ans)

                if correct:
                    correct_num += 1
                total += 1

        return correct_num, total, cleanIDTrain, cleanLabelTrain, id_confid

    def generateCleanDataByCrowd2(self, taskIDList):
        cleanIDTrain = []
        cleanLabelTrain = []

        confidence_curr = 1
        total = 0
        correct_num = 0

        id_confid = {}

        for idx in taskIDList:
            orc_argv = ["all"]
            selected_oracle_IDs = self.selectWeakOracles(orc_argv)
            # answer aggregation
            ans_argv = ["MV"]
            agg_ans, confidence_curr, correct = self.answerAggregation(idx, selected_oracle_IDs, ans_argv)

            id_confid[idx] = confidence_curr

            if confidence_curr >= self.m_confid_threshold:
                cleanIDTrain.append(idx)
                cleanLabelTrain.append(agg_ans)

                if correct:
                    correct_num += 1
                total += 1

        return correct_num, total, cleanIDTrain, cleanLabelTrain, id_confid

    '''
        gather labels of samples from selected oracles
    '''
    def gatherAnsFromWeakOracles(self, sampleIDs, selected_oracle_IDs):
        response = {}

        for idx in sampleIDs:
            all_answers = {}
            feature = corpusObj.m_feature[idx].reshape(1, -1)
            # gt_label = corpusObj.m_label[idx]

            # transfer
            partOracleStart = 0
            if self.m_useTransfer:
                partOracleStart += len(corpusObj.m_transferLabelList)
                for i in range(len(corpusObj.m_transferLabelList)):
                    transferLabel = corpusObj.m_transferLabelList[i][idx]
                    all_answers[i] = [transferLabel]  # {transferID: label}

            # gather answers from selected oracles
            for i in selected_oracle_IDs:
                cur_woracle = self.m_weakOracles[i]
                cur_ans = cur_woracle.predict(feature)
                all_answers[i + partOracleStart] = [cur_ans[0]]  # {oracleID: label}

            # classifier
            if self.m_useClf:
                clfLabel = self.m_activeClf.predict(feature)
                all_answers[-1] = [clfLabel]  # {classifier: label}

            response[idx] = all_answers  # {sampleID: {oracleID: [label]}}

        return response

    '''
    use dawid-skene EM algorithm to estimate class prior, predicted answer and confusion matrix
    ATTENTION: selected_oracle_IDs and m_unlabeledIDList MUST be SORTED in increasing order!!!
    '''
    def generateCleanDataByCrowd_EM(self, corpusObj, sampleIDs, fix_answers):

        cleanIDTrain = []
        cleanLabelTrain = []

        total = 0
        correct_num = 0
        id_confid = {}

        sampleIDs.sort()

        selected_oracle_IDs = [i for i in range(len(self.m_weakOracles))]
        responses = self.gatherAnsFromWeakOracles(sampleIDs, selected_oracle_IDs)

        #fix_answers = {id: corpusObj.m_label[id] for id in sampleIDs}
        classes_list = set(corpusObj.m_label2Idx.keys())

        #if self.m_prior_pred is None and self.m_weakOracleCM_pred is None:
        self.m_prior_pred, self.m_weakOracleCM_pred, ans_distri = DSEM.run(responses, fix_answers, init_classes=classes_list)  # no initial
        #else:
            #self.m_prior_pred, self.m_weakOracleCM_pred, ans_distri = DSEM.run(responses, fix_answers, self.m_prior_true, self.m_weakOracleCM_true)  # initial

        if len(fix_answers) % 50 == 4:
            print("class prior pred: "+str(self.m_prior_pred))

        # compute acc from CM, and RMSE
        acc_list = self.computeAccWithCM(self.m_weakOracleCM_pred, self.m_prior_pred)

        if self.m_useTransfer:
            self.m_weakOracleAcc_pred = acc_list[len(corpusObj.m_transferLabelList):]  # !!! revise if not all oracle are used
            self.m_accTransfer_pred = acc_list[:len(corpusObj.m_transferLabelList)]
            self.m_rmsePredWeakAcc.append(
                self.computeRMSEofAcc(acc_list, corpusObj.m_accTransferList + self.m_weakOracleAcc_true))
        '''
        else:
            self.m_weakOracleAcc_pred = acc_list  # !!! revise if not all oracle are used
            self.m_rmsePredWeakAcc.append(
                self.computeRMSEofAcc(self.m_weakOracleAcc_pred, self.m_weakOracleAcc_true))
        '''
        if len(fix_answers) % 50 == 4:
            print("current weak oracle acc prediction: " + str(acc_list))

        for i in range(len(ans_distri)):
            idx = sampleIDs[i]
            gt_label = corpusObj.m_label[idx]

            ans_i = list(ans_distri[i])
            confidence_curr = max(ans_i) / sum(ans_i)
            id_confid[idx] = confidence_curr
            agg_ans = corpusObj.m_idx2Label[ans_i.index(max(ans_i))]

            correct = gt_label == agg_ans

            if confidence_curr >= self.m_confid_threshold:
                cleanIDTrain.append(idx)
                cleanLabelTrain.append(agg_ans)

                if correct:
                    correct_num += 1
                total += 1

        return correct_num, total, cleanIDTrain, cleanLabelTrain, id_confid, responses


    def computeAccWithCM(self, CMs, prior):
        acc_list = []

        for i in range(len(CMs)):  # for each weak oracle
            CM_i = CMs[i]
            acc_i = 0.0
            for m in range(len(CM_i)):  # for each row
                ioe_row_m = [c * prior[m] for c in CM_i[m]]  # row m in Incidence-of-error probability matrix
                acc_i += ioe_row_m[m]  # sum of diagonal elements of IOE
            acc_list.append(acc_i)

        return acc_list

    def computeRMSEofAcc(self, acc_pred, acc_true):
        RMSE = 0.0
        for i in range(len(acc_pred)):
            RMSE += math.pow(acc_pred[i] - acc_true[i], 2)
        RMSE = math.sqrt(RMSE / len(acc_pred))
        # print("RMSE: " + str(RMSE))
        return RMSE

    def computeRMSEofCM(self, CMs, CMs_true):
        RMSE = 0.0
        count = 0
        for i in range(len(CMs)):
            CM_i = CMs[i]
            CM_true_i = CMs_true[i]
            for m in range(len(CM_i)):
                for n in range(CM_i[m]):
                    RMSE += math.pow(CM_i[m][n] - CM_true_i[m][n], 2)
                    count += 1
        RMSE = math.sqrt(RMSE/count)
        return RMSE

    def estimateOracleParam(self, goldIDs):
        acc_oracle_pred = []
        nObservers = len(self.m_weakOracles)
        for i in range(len(self.m_weakOracles)):
            acc = self.m_weakOracles[i].score(corpusObj.m_feature[goldIDs], corpusObj.m_label[goldIDs])
            acc_oracle_pred.append(acc)
        self.m_weakOracleAcc_pred = acc_oracle_pred

        # calculate the RMSE for estimation of weak oracle accuracy
        RMSE = 0.0
        for i in range(len(self.m_weakOracleAcc_pred)):
            RMSE += math.pow(self.m_weakOracleAcc_pred[i] - self.m_weakOracleAcc_true[i], 2)
        if self.m_useTransfer:
            nObservers += len(corpusObj.m_transferLabelList)
            for i in range(len(corpusObj.m_transferLabelList)):
                self.m_accTransfer_pred[i] = accuracy_score(corpusObj.m_label[goldIDs], corpusObj.m_transferLabelList[i][goldIDs])
                RMSE += math.pow(corpusObj.m_accTransferList[i] - self.m_accTransfer_pred[i], 2)

        RMSE = math.sqrt(RMSE / nObservers)
        # print("RMSE: " + str(RMSE))
        self.m_rmsePredWeakAcc.append(RMSE)

    def update_tao(self, labeled_set, corpusObj):

        dist_inter = []
        pair = list(itertools.combinations(labeled_set,2))

        for p in pair:

            d = np.linalg.norm(corpusObj.m_feature[p[0]]-corpusObj.m_feature[p[1]])
            if corpusObj.m_label[p[0]] != corpusObj.m_label[p[1]]:
                dist_inter.append(d)

        try:
            self.tao = self.alpha_*min(dist_inter)/2 #set tao be the min(inter-class pair dist)/2
        except Exception as e:
            self.tao = self.tao


    def update_pseudo_set(self, new_ex_id, cluster_id, p_idx, p_label, p_dist, corpusObj):

        tmp = []
        idx_tmp=[]
        label_tmp=[]

        #re-visit exs removed on previous itr with the new tao
        for i,j in zip(p_idx,p_label):

            if p_dist[i] < self.tao:
                idx_tmp.append(i)
                label_tmp.append(j)
            else:
                p_dist.pop(i)
                tmp.append(i)

        p_idx = idx_tmp
        p_label = label_tmp

        #added exs to pseudo set
        for ex in self.ex_id[cluster_id]:

            if ex == new_ex_id:
                continue
            d = np.linalg.norm(corpusObj.m_feature[ex]-corpusObj.m_feature[new_ex_id])

            if d < self.tao:
                p_dist[ex] = d
                p_idx.append(ex)
                p_label.append(corpusObj.m_label[new_ex_id])
            else:
                tmp.append(ex)

        if not tmp:
            self.ex_id.pop(cluster_id)
        else:
            self.ex_id[cluster_id] = tmp

        return p_idx, p_label, p_dist


    def select_example_cluster(self, labeled_set, corpusObj, clf = None):
        if clf is None:
            clf = self.clf

        sub_pred = dd(list) #Mn predicted labels for each cluster
        idx = 0
        # initialize: find the first instance which is not in labeled dataset
        for id in self.m_train:
            if id not in labeled_set:
                idx = id
                break

        for k,v in self.ex_id.items():
            sub_pred[k] = clf.predict(corpusObj.m_feature[v]) #predict labels for cluster learning set

        #entropy-based cluster selection
        rank = []
        for k,v in sub_pred.items():
            count = ct(v).values()
            count[:] = [i/float(max(count)) for i in count]
            H = np.sum(-p*math.log(p,2) for p in count if p!=0)
            rank.append([k,len(v),H])
        rank = sorted(rank, key=lambda x: x[-1], reverse=True)

        if not rank:
            raise ValueError('no clusters found in this iteration!')

        c_idx = rank[0][0] #pick the 1st cluster on the rank, ordered by label entropy
        c_ex_id = self.ex_id[c_idx] #examples in the cluster picked
        sub_label = sub_pred[c_idx] #used when choosing cluster by H
        sub_fn = corpusObj.m_feature[c_ex_id]

        #sub-cluster the cluster
        c_ = KMeans(init='k-means++', n_clusters=len(np.unique(sub_label)), n_init=10)
        c_.fit(sub_fn)
        dist = np.sort(c_.transform(sub_fn))

        ex_ = dd(list) # clusterid, id, dist, predicted label
        for i,j,k,l in zip(c_.labels_, c_ex_id, dist, sub_label):
            ex_[i].append([j,l,k[0]])
        for i,j in ex_.items(): #sort by ex. dist to the centroid for each C
            ex_[i] = sorted(j, key=lambda x: x[-1])
        for k,v in ex_.items():

            if v[0][0] not in labeled_set: #find the first unlabeled ex

                idx = v[0][0]

                c_ex_id.remove(idx) #update the training set by removing selected ex id

                if len(c_ex_id) == 0:
                    self.ex_id.pop(c_idx)
                else:
                    self.ex_id[c_idx] = c_ex_id
                break

        '''
        self.ex_id.pop(c_idx)
        new_clusterid = max(self.ex_id) + 1
        uniq_sub_lbs = np.unique(c_.labels_)
        for i in range(len(c_.labels_)):
            cid = new_clusterid + i
            tmp = []
            for lb, id in zip(c_.labels_, c_ex_id):
                if lb == uniq_sub_lbs[i]:
                    tmp.append(id)
            self.ex_id[cid] = tmp
        '''

        return idx, c_idx

    def sub_cluster(self, cid, clf):
        if cid not in self.ex_id:
            return

        c_ex_id = self.ex_id[cid]
        sub_label = clf.predict(corpusObj.m_feature[c_ex_id])
        sub_fn = corpusObj.m_feature[c_ex_id]

        c_ = KMeans(init='k-means++', n_clusters=len(np.unique(sub_label)), n_init=10)
        c_.fit(sub_fn)
        dist = np.sort(c_.transform(sub_fn))  # dist[i][j], i-th instance, j-th closest distance

        self.ex_id.pop(cid)

        new_clusterid = max(self.ex_id) + 1
        uniq_sub_lbs = np.unique(c_.labels_)

        sub_dict = {lb: [] for lb in uniq_sub_lbs}
        for lb, id in zip(c_.labels_, c_ex_id):
            sub_dict[lb].append(id)
        for i in range(len(uniq_sub_lbs)):
            cid = new_clusterid + i
            self.ex_id[cid] = sub_dict[i]
        return


    def revise_cluster(self, labeled_set, p_idx, p_label):
        ex_id_cs_new = dd(list)
        new_clusters = []
        p_dict = {p_idx[i]: p_label[i] for i in range(len(p_idx))}
        for cid in self.ex_id_cs:
            lb_uniq = {}
            for id in self.ex_id_cs[cid]:
                if id in labeled_set:
                    lb = corpusObj.m_label[id] # labeled
                else:
                    if id in p_dict:
                        lb = p_dict[id]  # propagated
                    else:
                        lb = self.clf.predict([corpusObj.m_feature[id]])[0]
                if lb not in lb_uniq:
                    lb_uniq[lb] = [id]
                else:
                    lb_uniq[lb].append(id)
            for lb in lb_uniq:
                new_clusters.append(lb_uniq[lb])

        for i, ids_c in enumerate(new_clusters):
            ex_id_cs_new[i] = ids_c
        id2cluster_new = {id: cid for cid in ex_id_cs_new for id in ex_id_cs_new[cid]}

        self.ex_id_cs = ex_id_cs_new
        self.id2cluster = id2cluster_new

        return

    def revise_cluster_AL(self, labeled_set, p_idx, p_label):
        ex_id_new = dd(list)
        new_clusters = []
        p_dict = {p_idx[i]: p_label[i] for i in range(len(p_idx))}
        for cid in self.ex_id:
            lb_uniq = {}
            for id in self.ex_id[cid]:
                if id in labeled_set:
                    lb = corpusObj.m_label[id] # labeled
                else:
                    if id in p_dict:
                        lb = p_dict[id]  # propagated
                    else:
                        lb = self.clf.predict([corpusObj.m_feature[id]])[0]
                if lb not in lb_uniq:
                    lb_uniq[lb] = [id]
                else:
                    lb_uniq[lb].append(id)
            for lb in lb_uniq:
                new_clusters.append(lb_uniq[lb])

        for i, ids_c in enumerate(new_clusters):
            ex_id_new[i] = ids_c

        self.ex_id = ex_id_new

        return


    def select_example_weak(self, labeled_set, corpusObj, weakAns_dict, clf = None):
        if clf is None:
            clf = self.clf
        clf_al = self.clf_al

        # disagreement between clf_al and weak answers
        # choose the one with most neighbors
        dis_id = {} # {disagreed id: neighbor num}
        for id in self.m_train:
            if id in labeled_set:
                continue
            cluster_id = self.id2cluster[id]  # cluster id in crowdsourcing (or AL?)

            lb_pre_al = clf_al.predict([corpusObj.m_feature[id]])
            lb_pre = clf.predict([corpusObj.m_feature[id]])
            if lb_pre_al[0] != lb_pre[0]:
                # count num of neighbors
                for ex in self.ex_id_cs[cluster_id]:
                    if ex == id:
                        continue
                    d = np.linalg.norm(corpusObj.m_feature[ex] - corpusObj.m_feature[id])
                    if d < self.tao:
                        if id not in dis_id:
                            dis_id[id] = 0
                        else:
                            dis_id[id] += 1

        id2c = {id: cid for cid in self.ex_id for id in self.ex_id[cid]} # cluster for AL
        if len(dis_id) > 0:
            sort_dis_id = sorted(dis_id.items(), key=lambda x: x[1], reverse=True)
            idx = sort_dis_id[0][0]
            c_idx = id2c[idx]

            # remove idx from cluster of AL
            c_ex_id = self.ex_id[c_idx]
            c_ex_id.remove(idx)  # update the training set by removing selected ex id

            if len(c_ex_id) == 0:
                self.ex_id.pop(c_idx)
            else:
                self.ex_id[c_idx] = c_ex_id

        else:
            idx, c_idx = self.select_example_cluster(labeled_set, corpusObj)

        return idx, c_idx


    def get_pred_acc(self, fn_test, label_test, labeled_set, pseudo_set, pseudo_label, weakAns_dict, corpusObj, clf = None):
        '''
        # overlap in label and propagated set?
        intersect_l_p = [id for id in labeled_set if id in pseudo_set]
        if len(intersect_l_p)>0:
            print()
        '''
        if clf is None:
            clf = self.clf

        if not pseudo_set:
            trn_iter_id = labeled_set
            trn_iter_label = corpusObj.m_label[labeled_set]
        else:
            trn_iter_id = np.hstack((labeled_set, pseudo_set))
            trn_iter_label = np.hstack((corpusObj.m_label[labeled_set], pseudo_label))

        # weak answers
        wid_list = []
        wlabel_list = []
        for id in weakAns_dict:
            labelID_set = set(labeled_set).union(set(pseudo_set)) # set of labeled and pseudo-labeled ID
            if id not in labelID_set:
                wid_list.append(id)
                wlabel_list.append(weakAns_dict[id])
        if wid_list:
            trn_iter_id = np.hstack((trn_iter_id, wid_list))
            trn_iter_label = np.hstack((trn_iter_label, wlabel_list))

        fn_train = corpusObj.m_feature[trn_iter_id]

        clf.fit(fn_train, trn_iter_label)
        fn_preds = clf.predict(fn_test)

        acc = accuracy_score(label_test, fn_preds)
        # print("acc\t", acc)
        correctNum = 0
        for i in range(len(trn_iter_id)):
            id = trn_iter_id[i]
            if trn_iter_label[i] == corpusObj.m_label[id]:
                correctNum += 1
        return acc, len(fn_train), correctNum

    def addWeakAns(self, labeled_set, pseudo_set, pseudo_label, weakAns_dict, corpusObj):
        # label + pseudo
        if not pseudo_set:
            trn_iter_id = labeled_set
            trn_iter_label = corpusObj.m_label[labeled_set]
        else:
            trn_iter_id = np.hstack((labeled_set, pseudo_set))
            trn_iter_label = np.hstack((corpusObj.m_label[labeled_set], pseudo_label))

        # weak answers
        wid_list = []
        wlabel_list = []
        for id in weakAns_dict:
            labelID_set = set(labeled_set).union(set(pseudo_set))  # set of labeled and pseudo-labeled ID
            if id not in labelID_set:
                wid_list.append(id)
                wlabel_list.append(weakAns_dict[id])
        if wid_list:
            trn_iter_id = np.hstack((trn_iter_id, wid_list))
            trn_iter_label = np.hstack((trn_iter_label, wlabel_list))

        fn_train = corpusObj.m_feature[trn_iter_id]
        return fn_train, trn_iter_label

    # cluster-based confusion matrix
    # cid_ex: [clusterID:{exID1, exID2,...}]
    def aggregateWeakAns_cluster(self, weak_response, fix_answers, corpusObj, cid_list = None):
        classes_list = set(corpusObj.m_label2Idx.keys())
        ans_pred_cluster = {}
        correct_num = 0
        total = 0

        if cid_list is None:
            cid_list = self.ex_id_cs
        for cid in cid_list:
            ex_ids_c = self.ex_id_cs[cid]
            ex_ids_c_set = set(ex_ids_c)
            responses = {id: weak_response[id] for id in ex_ids_c_set}
            fix_answers_c = {id: fix_answers[id] for id in fix_answers if id in ex_ids_c_set}

            prior_pred_c, weakOracleCM_pred_c, ans_distri = DSEM.run(responses, fix_answers_c,
                                                                   init_classes=classes_list)  # no initial
            self.m_weakOracleCM_cluster_pred[cid] = weakOracleCM_pred_c # {cluster:{oracle: CM}}

            # compute acc from CM, and RMSE
            acc_list = self.computeAccWithCM(weakOracleCM_pred_c, prior_pred_c)

            if self.m_useTransfer:
                self.m_weakOracleAcc_cluster_pred[cid] = acc_list[len(
                    corpusObj.m_transferLabelList):]  # !!! revise if not all oracle are used
                self.m_accTransfer_cluster_pred[cid] = acc_list[:len(corpusObj.m_transferLabelList)]
                #self.m_rmsePredWeakAcc.append(
                #    self.computeRMSEofAcc(acc_list, corpusObj.m_accTransferList + self.m_weakOracleAcc_true))


            sampleIDs = ex_ids_c
            sampleIDs.sort()

            # answer
            for i in range(len(ans_distri)):
                idx = sampleIDs[i]
                gt_label = corpusObj.m_label[idx]

                ans_i = list(ans_distri[i])
                confidence_curr = max(ans_i) / sum(ans_i)
                #id_confid[idx] = confidence_curr
                agg_ans = corpusObj.m_idx2Label[ans_i.index(max(ans_i))]

                correct = gt_label == agg_ans

                if confidence_curr >= 0.9: #self.m_confid_threshold:
                    ans_pred_cluster[idx] = agg_ans

                    if correct:
                        correct_num += 1
                    total += 1
        return ans_pred_cluster, correct_num, total

    def getfixAns(self, labeled_list, pseudo_list, pseudo_label):
        fix_ans = {}

        if not pseudo_list:
            trn_iter_id = labeled_list
            trn_iter_label = corpusObj.m_label[labeled_list]
        else:
            labeled_set = set(labeled_list)
            pseudo_set_clean_idx = [i for i in range(len(pseudo_list)) if pseudo_list[i] not in labeled_set]
            pseudo_set_clean = np.array(pseudo_list)[pseudo_set_clean_idx]
            pseudo_label_clean = np.array(pseudo_label)[pseudo_set_clean_idx]
            trn_iter_id = np.hstack((labeled_list, pseudo_set_clean))
            trn_iter_label = np.hstack((corpusObj.m_label[labeled_list], pseudo_label_clean))

        for i in range(len(trn_iter_id)):
            fix_ans[trn_iter_id[i]] = trn_iter_label[i]
        return fix_ans


    def activeTrainClf_al(self, corpusObj):
        # clustering
        c = KMeans(init='k-means++', n_clusters=26, n_init=10)
        c.fit(corpusObj.m_feature[self.m_train])
        dist = np.sort(c.transform(corpusObj.m_feature[self.m_train]))  # size: instanceNum x clusterNum

        ex = dd(list)  # example id, distance to centroid # {clusterID: [[idx, dist],...]}
        self.ex_id = dd(list)  # example id for each C  # {clusterID: [idxs]}
        ex_N = []  # num of examples in each C  #[[clusterID, size]]
        for i, j, k in zip(c.labels_, self.m_train, dist):
            ex[i].append([j, k[0]])
            self.ex_id[i].append(int(j))
        for i, j in ex.items():
            ex[i] = sorted(j, key=lambda x: x[-1])
            ex_N.append([i, len(ex[i])])
        ex_N = sorted(ex_N, key=lambda x: x[-1], reverse=True)

        km_idx = []
        p_idx = []
        p_label = []
        p_dist = dd()
        # first batch of exs: pick centroid of each cluster, and cluster visited based on its size
        ctr = 0
        for ee in ex_N:

            c_idx = ee[0]  # cluster id
            idx = ex[c_idx][0][0]  # id of ex closest to centroid of cluster
            km_idx.append(idx)
            ctr += 1

            if ctr < 3:  # \ get at least instances for initialization
                continue

            self.update_tao(km_idx, corpusObj)

            p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist, corpusObj)

            try:
                acc = self.get_pred_acc(corpusObj.m_feature[self.m_test], corpusObj.m_label[self.m_test], km_idx, p_idx,
                                        p_label, dict(), corpusObj)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print exc_type, e.args, fname, 'on line ' + str(exc_tb.tb_lineno)
                acc = np.nan

            self.m_accList.append(acc)
            # print("acc\t", acc)

        cl_id = []  # track cluster id on each iter
        ex_al = []  # track ex added on each iter
        fn_test = corpusObj.m_feature[self.m_test]
        label_test = corpusObj.m_label[self.m_test]
        for rr in range(ctr, self.m_strongLabelNumThresh):
            '''
            if not p_idx:
                fn_train = corpusObj.m_feature[km_idx]
                label_train = corpusObj.m_label[km_idx]
            else:
                fn_train = corpusObj.m_feature[np.hstack((km_idx, p_idx))]
                label_train = np.hstack((corpusObj.m_label[km_idx], p_label))
            '''
            fn_train, label_train = self.addWeakAns(km_idx, p_idx, p_label, dict(), corpusObj)
            self.clf.fit(fn_train, label_train)

            idx, c_idx, = self.select_example_cluster(km_idx, corpusObj)
            #idx, c_idx, = self.select_example_weak(km_idx, corpusObj, {})
            km_idx.append(idx)
            cl_id.append(c_idx)  # track picked cluster id on each iteration
            # ex_al.append([rr,key,v[0][-2],corpusObj.m_label[idx],raw_pt[idx]]) #for debugging

            self.update_tao(km_idx, corpusObj)
            p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist, corpusObj)

            acc = self.get_pred_acc(fn_test, label_test, km_idx, p_idx, p_label, dict(), corpusObj)
            self.m_accList.append(acc)

            # sub-cluster the cluster
            self.sub_cluster(c_idx, self.clf)

            if rr % 20 == 0:
                print(rr, acc)
                #print(rr, len(self.ex_id))

        #print("finished!")

    def activeTrainClf_global(self, corpusObj):
        # weak oracle answers
        strongAnswers_dict = {}
        self.m_train.sort()

        correct, total, cleanIDTrain, cleanLabelTrain, id_conf, responses = self.generateCleanDataByCrowd_EM(corpusObj,self.m_train,
                                                                                                             dict(strongAnswers_dict))
        correct_num, total, cleanIDTrain, cleanLabelTrain, id_conf = self.generateCleanDataByCrowd2(self.m_train)

        cleanFeatureTrain = corpusObj.m_feature[cleanIDTrain]
        weakAnswers_dict_global = {cleanIDTrain[i]: cleanLabelTrain[i] for i in range(len(cleanIDTrain))}

        # confidence - accuracy
        '''
        cf_thresh = [0.5, 0.6, 0.8, 0.86, 0.93, 0.94, 0.95, 0.96,0.97,0.98,0.99]
        total_cf = [0 for i in range(len(cf_thresh))]
        correct_cf = [0 for i in range(len(cf_thresh))]
        for id in id_conf:
            if id_conf[id] < self.m_confid_threshold:
                continue
            correct = corpusObj.m_label[id] == weakAnswers_dict_global[id]
            for cf_i in range(len(cf_thresh)):
                cfs = cf_thresh[cf_i]
                if id_conf[id] >= cfs:
                    total_cf[cf_i] += 1
                    if correct:
                        correct_cf[cf_i] += 1
        acc_cf = [float(correct_cf[i]) / total_cf[i] for i in range(len(cf_thresh)) if total_cf[i] > 0]
        print(correct_cf, total_cf)
        print(acc_cf)
        '''

        # clustering
        c = KMeans(init='k-means++', n_clusters=26, n_init=10)
        c.fit(corpusObj.m_feature[self.m_train])
        dist = np.sort(c.transform(corpusObj.m_feature[self.m_train]))  # size: instanceNum x clusterNum

        ex = dd(list)  # example id, distance to centroid # {clusterID: [[idx, dist],...]}
        self.ex_id = dd(list)  # example id for each C  # {clusterID: [idxs]}
        ex_N = []  # num of examples in each C  #[[clusterID, size]]
        for i, j, k in zip(c.labels_, self.m_train, dist):
            ex[i].append([j, k[0]])
            self.ex_id[i].append(int(j))
        for i, j in ex.items():
            ex[i] = sorted(j, key=lambda x: x[-1])
            ex_N.append([i, len(ex[i])])
        ex_N = sorted(ex_N, key=lambda x: x[-1], reverse=True)

        km_idx = []
        p_idx = []
        p_label = []
        p_dist = dd()
        # first batch of exs: pick centroid of each cluster, and cluster visited based on its size
        ctr = 0
        for ee in ex_N:

            c_idx = ee[0]  # cluster id
            idx = ex[c_idx][0][0]  # id of ex closest to centroid of cluster
            km_idx.append(idx)
            ctr += 1

            if ctr < 3:  # \ get at least instances for initialization
                continue

            self.update_tao(km_idx, corpusObj)

            p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist, corpusObj)

            try:
                acc = self.get_pred_acc(corpusObj.m_feature[self.m_test], corpusObj.m_label[self.m_test], km_idx, p_idx,
                                        p_label, weakAnswers_dict_global, corpusObj)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print exc_type, e.args, fname, 'on line ' + str(exc_tb.tb_lineno)
                acc = np.nan

            self.m_accList.append(acc)
            # print("acc\t", acc)

        cl_id = []  # track cluster id on each iter
        ex_al = []  # track ex added on each iter
        fn_test = corpusObj.m_feature[self.m_test]
        label_test = corpusObj.m_label[self.m_test]
        for rr in range(ctr, self.m_strongLabelNumThresh):
            '''
            if not p_idx:
                fn_train = corpusObj.m_feature[km_idx]
                label_train = corpusObj.m_label[km_idx]
            else:
                fn_train = corpusObj.m_feature[np.hstack((km_idx, p_idx))]
                label_train = np.hstack((corpusObj.m_label[km_idx], p_label))
            '''
            fn_train, label_train = self.addWeakAns(km_idx, p_idx, p_label, weakAnswers_dict_global, corpusObj)
            self.clf.fit(fn_train, label_train)

            idx, c_idx, = self.select_example_cluster(km_idx, corpusObj, self.clf_al)

            km_idx.append(idx)
            cl_id.append(c_idx)  # track picked cluster id on each iteration
            # ex_al.append([rr,key,v[0][-2],corpusObj.m_label[idx],raw_pt[idx]]) #for debugging

            self.update_tao(km_idx, corpusObj)
            p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist, corpusObj)

            acc = self.get_pred_acc(fn_test, label_test, km_idx, p_idx, p_label, weakAnswers_dict_global, corpusObj)
            self.m_accList.append(acc)

            if rr % 20 == 0:
                print(rr, acc)

        print("finished!")

    def filter_from_cluster(self, dataset, ex_id_cs, id2cluster, responses, c_res_num, rate):
        filtered_id = []
        for id in dataset:
            res_i = str(responses[id])
            cid = id2cluster[id]
            cid_size = len(ex_id_cs[cid])
            if c_res_num[cid][res_i] >= rate * cid_size:
                filtered_id.append(id)
        return filtered_id

    def computeTrueCM(self, dataset):
        nObservers = len(self.m_weakOracles)
        partOracleStart = 0
        if self.m_useTransfer:
            nObservers += len(corpusObj.m_transferLabelList)
            partOracleStart += len(corpusObj.m_transferLabelList)

        error_rates = np.zeros([nObservers, corpusObj.m_labelNum, corpusObj.m_labelNum])

        # CM of transfer
        if self.m_useTransfer:
            untrn = dataset
            untrn_features = corpusObj.m_feature[untrn]
            untrn_labels = corpusObj.m_label[untrn]

            for i in range(len(corpusObj.m_transferLabelList)):
                untrn_labels_pred = corpusObj.m_transferLabelList[i][untrn]
                # n00, n01, n10, n11 = self.tool(untrn_labels, untrn_labels_pred)

                for j in range(len(untrn)):
                    label_idx_pred = corpusObj.m_label2Idx[untrn_labels_pred[j]]
                    label_idx_true = corpusObj.m_label2Idx[untrn_labels[j]]
                    error_rates[i][label_idx_true][label_idx_pred] += 1

                for j in range(corpusObj.m_labelNum):
                    sum_over_row = np.sum(error_rates[i, j, :])
                    if sum_over_row > 0:
                        error_rates[i, j, :] = error_rates[i, j, :] / float(sum_over_row)

        # CM of other oracles
        for i in range(len(self.m_weakOracles)):
            if self.m_useTransfer:
                oracle_id = i + partOracleStart
            else:
                oracle_id = i
            # acc on untrained data
            untrn = dataset
            untrn_features = corpusObj.m_feature[untrn]
            untrn_labels = corpusObj.m_label[untrn]
            untrn_labels_pred = self.m_weakOracles[i].predict(untrn_features)
            #n00, n01, n10, n11 = self.tool(untrn_labels, untrn_labels_pred)

            for j in range(len(untrn)):
                label_idx_pred = corpusObj.m_label2Idx[untrn_labels_pred[j]]
                label_idx_true = corpusObj.m_label2Idx[untrn_labels[j]]

                error_rates[oracle_id][label_idx_true][label_idx_pred] += 1

            for j in range(corpusObj.m_labelNum):
                sum_over_row = np.sum(error_rates[oracle_id, j, :])
                if sum_over_row > 0:
                    error_rates[oracle_id, j, :] = error_rates[oracle_id, j, :] / float(sum_over_row)

        return error_rates

    def computeClassPrior(self, dataset):
        label_train = corpusObj.m_label[dataset]
        labelNum_train = {}
        for lb in corpusObj.m_label2Idx:
            lb_num = list(label_train).count(lb)
            labelNum_train[lb] = lb_num
        for lb in labelNum_train:
            labelNum_train[lb] = float(labelNum_train[lb]) / len(dataset)
        prior_true = [labelNum_train[i] for i in sorted(labelNum_train.iterkeys())]
        return prior_true

    def computeRMSE_2D(self, matrix1, matrix2):
        RMSE = 0.0
        count = 0
        for i in range(len(matrix1)):
            for j in range(len(matrix1[i])):
                RMSE += math.pow(matrix1[i][j] - matrix2[i][j], 2)
                count += 1
        RMSE = math.sqrt(RMSE / count)
        return RMSE

    def experiment_111(self, responses):
        # < Experiment 1: Fix confusion matrix, class prior, run E-step
        # true CM and class prior on training dataset
        oracle_trueCM = self.computeTrueCM(self.m_train)
        class_prior_true = self.computeClassPrior(self.m_train)
        # true T*
        ans_distri_true = np.zeros([len(self.m_train), len(corpusObj.m_label2Idx)])
        for i in range(len(self.m_train)):
            idx = self.m_train[i]
            lb = corpusObj.m_label[idx]
            lb_id = corpusObj.m_label2Idx[lb]
            ans_distri_true[i][lb_id] = 1.0

        # E-step T'
        classes_list = set(corpusObj.m_label2Idx.keys())
        ans_distri = DSEM.runE_fixCM(responses, {}, class_prior_true, oracle_trueCM, classes_list)

        # answer
        correct_num = 0
        total = 0
        for i in range(len(ans_distri)):
            idx = self.m_train[i]
            gt_label = corpusObj.m_label[idx]

            ans_i = list(ans_distri[i])
            confidence_curr = max(ans_i) / sum(ans_i)
            # id_confid[idx] = confidence_curr
            agg_ans = corpusObj.m_idx2Label[ans_i.index(max(ans_i))]

            correct = gt_label == agg_ans

            if confidence_curr >= 0.0:  # self.m_confid_threshold:
                if correct:
                    correct_num += 1
                total += 1

        # print(correct_num, total)

        # RMSE between T' and T*
        e1_rmse = self.computeRMSE_2D(ans_distri, ans_distri_true)
        print(e1_rmse)

        # experiment 1/>

        '''
        # <Experiment 2: fix x%
        # true T*
        ans_distri_true = np.zeros([len(self.m_train), len(corpusObj.m_label2Idx)])
        for i in range(len(self.m_train)):
            idx = self.m_train[i]
            lb = corpusObj.m_label[idx]
            lb_id = corpusObj.m_label2Idx[lb]
            ans_distri_true[i][lb_id] = 1.0

        rate_x_lit = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        e2_rmse = []
        acc_wk_sample = []
        for rate_x in rate_x_lit:
            sampled_ids = random.sample(set(self.m_train), int(rate_x * len(self.m_train)))
            fix_samples = {id: corpusObj.m_label[id] for id in sampled_ids}

            classes_list = set(corpusObj.m_label2Idx.keys())
            prior_pred, weakOracleCM_pred, ans_distri = DSEM.run(responses, fix_samples, init_classes=classes_list)

            e2_rmse_x = self.computeRMSE_2D(ans_distri, ans_distri_true)
            e2_rmse.append(e2_rmse_x)

            correct_num = 0
            total = 0
            for i in range(len(ans_distri)):
                idx = self.m_train[i]
                gt_label = corpusObj.m_label[idx]

                ans_i = list(ans_distri[i])
                confidence_curr = max(ans_i) / sum(ans_i)
                # id_confid[idx] = confidence_curr
                agg_ans = corpusObj.m_idx2Label[ans_i.index(max(ans_i))]

                correct = gt_label == agg_ans

                if confidence_curr >= 0.0:  # self.m_confid_threshold:
                    if correct:
                        correct_num += 1
                    total += 1
            acc_wk_sample.append(float(correct_num)/total)

            #print(e2_rmse)
            #print("acc: ", acc_wk_sample)

        print(e2_rmse)
        print("acc:", acc_wk_sample)
        '''

        #
        # <Experiment 3: ideal clustering (by class) + EM
        # idea clustering

        ex_id_cs_cls = dd(list)
        id2cluster_cls = {}

        for id in self.m_train:
            lb = corpusObj.m_label[id]
            cid = corpusObj.m_label2Idx[lb]

            ex_id_cs_cls[cid].append(id)
            id2cluster_cls[id] = cid

        # self.ex_id_cs = ex_id_cs_cls
        # self.id2cluster = id2cluster_cls

        # clustering by feature
        feature1 = corpusObj.m_feature[self.m_train]
        feature2 = np.array([[responses[id][orc][0] for orc in responses[id]] for id in self.m_train])  # response
        # one-hot
        onehot_feature = []
        for i in range(len(feature2)):  # each instance
            onehot_feature_i = []
            for j in range(len(feature2[i])):  # eache oracle
                onehotlb = [0 for o in range(len(corpusObj.m_label2Idx))]
                res = feature2[i][j]
                onehotlb[corpusObj.m_label2Idx[res]] = 1
                onehot_feature_i.extend(onehotlb)
            onehot_feature.append(onehot_feature_i)
        feature2 = np.array(onehot_feature)  # * 100
        #
        feature1 = np.append(feature1, feature2, axis=1)

        ex_id_cs1, id2cluster1 = self.clusterWithFeatures(4, self.m_train, feature1)

        self.ex_id_cs = ex_id_cs1
        self.id2cluster = id2cluster1

        # global EM
        '''
        correct_num_global_EM = {} # global EM
        correct_num_global_2 = {} # global WV/BV +EM
        for cid in ex_id_cs_cls:
            correct_num = 0
            correct_num2 = 0
            for id in ex_id_cs_cls[cid]:
                lb = weakAnswers_dict_global[id]
                lb2 = weakAnswers_dict_global_2[id]
                if lb == corpusObj.m_label[id]:
                    correct_num += 1
                if lb2 == corpusObj.m_label[id]:
                    correct_num2 += 1
            correct_num_global_EM[cid] = (correct_num, len(ex_id_cs_cls[cid]))
            correct_num_global_2[cid] = (correct_num2, len(ex_id_cs_cls[cid]))
        '''
        # cluster EM
        fix_answers = {}
        correct_num_cluster = {}

        ans_pred_cluster, correct_num, total = self.aggregateWeakAns_cluster(responses, fix_answers, corpusObj)
        for cid in ex_id_cs_cls:
            correct_num = 0
            for id in ex_id_cs_cls[cid]:
                lb = ans_pred_cluster[id]
                if lb == corpusObj.m_label[id]:
                    correct_num += 1
            correct_num_cluster[cid] = (correct_num, len(ex_id_cs_cls[cid]))

        # print(correct_num_global_EM)
        print(correct_num_cluster)

        # MV, WV, BV
        correct, total, cleanIDTrain, cleanLabelTrain, id_conf = self.generateCleanDataByCrowd_cluster(self.m_train)
        weakAnswers_dict = {cleanIDTrain[i]: cleanLabelTrain[i] for i in range(len(cleanIDTrain))}
        correct_num_cluster = {}
        for cid in ex_id_cs_cls:
            correct_num = 0
            for id in ex_id_cs_cls[cid]:
                lb = weakAnswers_dict[id]
                if lb == corpusObj.m_label[id]:
                    correct_num += 1
            correct_num_cluster[cid] = (correct_num, len(ex_id_cs_cls[cid]))

        # print(correct_num_global_2)
        print("cls:", correct_num_cluster)

        print(self.m_accTransfer_cluster_pred)

    def response_distri(self,responses):
        res_dic = {}
        for id in responses:
            res = str(responses[id])
            if res not in res_dic:
                res_dic[res] = 1
            else:
                res_dic[res] += 1
        return res_dic

    def classDistri(self, instances, corpusObj):
        cls_dict = {lb:0 for lb in corpusObj.m_label2Idx}
        for id in instances:
            cls_dict[corpusObj.m_label[id]] += 1
        return cls_dict

    def expeiment_1(self, corpusObj):
        trainSampleNum = []
        # weak oracle answers
        self.m_train.sort()

        responses = self.gatherAnsFromWeakOracles(self.m_train, [])
        '''
        # for each class
        resNum_per_class = {lb: {} for lb in corpusObj.m_label2Idx}
        res_sensorName = {lb: {} for lb in corpusObj.m_label2Idx}
        for id in self.m_train:
            res = str(responses[id])
            lb = corpusObj.m_label[id]
            nm = corpusObj.m_sensorName[id]
            if res not in resNum_per_class[lb]:
                resNum_per_class[lb][res] = 1
                res_sensorName[lb][res] = [nm]
            else:
                resNum_per_class[lb][res] += 1
                res_sensorName[lb][res].append(nm)
        print("============")
        print(res_sensorName)
        print("-----------")
        '''
        '''
        # weak with cluster
        # clustering: the cluster here is used by the cluster-based crowdsourcing
        feature1 = corpusObj.m_feature[self.m_train]
        feature2 = np.array([[responses[id][orc][0] for orc in responses[id]] for id in self.m_train])  # response
        # one-hot
        onehot_feature = []
        for i in range(len(feature2)):  # each instance
            onehot_feature_i = []
            for j in range(len(feature2[i])):  # eache oracle
                onehotlb = [0 for o in range(len(corpusObj.m_label2Idx))]
                res = feature2[i][j]
                onehotlb[corpusObj.m_label2Idx[res]] = 1
                onehot_feature_i.extend(onehotlb)
            onehot_feature.append(onehot_feature_i)
        feature2 = np.array(onehot_feature) * 100
        #
        # feature1 = np.append(feature1, feature2, axis=1)

        ex_id_cs1, id2cluster1 = self.clusterWithFeatures(28, self.m_train, feature1)

        # name
        cluster_lb_name = {cid:{} for cid in ex_id_cs1}
        for cid in ex_id_cs1:
            for id in ex_id_cs1[cid]:
                lb = corpusObj.m_label[id]
                nm = corpusObj.m_sensorName[id]
                if lb not in cluster_lb_name[cid]:
                    cluster_lb_name[cid][lb] = [nm]
                else:
                    cluster_lb_name[cid][lb].append(nm)

        # statistics
        self.ex_id_cs = ex_id_cs1
        self.id2cluster = id2cluster1

        # crowdsourcing by cluster
        fix_answers = {}
        # feature 1
        ans_pred_cluster, correct_num, total = self.aggregateWeakAns_cluster(responses, fix_answers, corpusObj)
        print(correct_num, total)
        cor_lb_dict = {lb: 0 for lb in corpusObj.m_label2Idx} # label: correct num
        cor_res_dict = {lb:{} for lb in corpusObj.m_label2Idx}
        for id in ans_pred_cluster:
            if ans_pred_cluster[id] == corpusObj.m_label[id]:
                cor_lb_dict[corpusObj.m_label[id]] += 1
                res = str(responses[id])
                if res in cor_res_dict[corpusObj.m_label[id]]:
                    cor_res_dict[corpusObj.m_label[id]][res] += 1
                else:
                    cor_res_dict[corpusObj.m_label[id]][res] = 1

        print(cor_lb_dict)
        print(cor_res_dict)


        # clustering
        c = KMeans(init='k-means++', n_clusters=28, n_init=10)
        c.fit(corpusObj.m_feature[self.m_train])
        dist = np.sort(c.transform(corpusObj.m_feature[self.m_train]))  # size: instanceNum x clusterNum

        ex = dd(list)  # example id, distance to centroid # {clusterID: [[idx, dist],...]}
        self.ex_id = dd(list)  # example id for each C  # {clusterID: [idxs]}
        ex_N = []  # num of examples in each C  #[[clusterID, size]]
        for i, j, k in zip(c.labels_, self.m_train, dist):
            ex[i].append([j, k[0]])
            self.ex_id[i].append(int(j))
        for i, j in ex.items():
            ex[i] = sorted(j, key=lambda x: x[-1])
            ex_N.append([i, len(ex[i])])
        ex_N = sorted(ex_N, key=lambda x: x[-1], reverse=True)

        lb_distr = {}
        for cid in self.ex_id:
            lbdistri_c = self.classDistri(self.ex_id[cid], corpusObj)
            lb_distr[cid] = lbdistri_c
        print("============")
        print(lb_distr)
        print("============")

        trainSampleNum = []
        # biased weak ans
        '''
        '''
        labels_I_need = {4.0, 6.0, 5.0} # bias
        sample_lb4 = [id for id in self.m_train if corpusObj.m_label[id] in labels_I_need]
        #sample_lb4 = random.sample(sample_lb4, int(len(sample_lb4) * 1.0))  # coverage
        wrong_x = 0.2 # acc
        wrong_ids = random.sample(sample_lb4, int(len(sample_lb4) * wrong_x))
        wrongAns_dict = {}
        for id in wrong_ids:
            labels = [lb for lb in corpusObj.m_label2Idx if corpusObj.m_label[id] != lb]
            wrong_lb = random.sample(labels, 1)[0]
            wrongAns_dict[id] = wrong_lb

        weakAnswers_dict = wrongAns_dict.copy()
        for id in sample_lb4:
            if id not in wrongAns_dict:
                weakAnswers_dict[id] = corpusObj.m_label[id]
        '''

        #response with high frequency
        '''
        
        responses = self.gatherAnsFromWeakOracles(self.m_train, [])


        response_feq = np.array([str(responses[id]) for id in responses])
        values, counts = np.unique(response_feq, return_counts=True)
        response_feq = dict(zip(values, counts))
        sorted_res = sorted(response_feq.items(), key=lambda kv: kv[1], reverse=True)

        top_k = 2
        top_res = {sorted_res[i][0] for i in range(top_k)}

        top_res_lbs = {res:{} for res in top_res} # res: lb: count
        for id in self.m_train:
            res = str(responses[id])
            if res not in top_res:
                continue
            lb = corpusObj.m_label[id]
            if lb not in top_res_lbs[res]:
                top_res_lbs[res][lb] = 1
            else:
                top_res_lbs[res][lb] += 1
        '''
        '''
        res_list = {'{0: [4.0], 1: [4.0], 2: [6.0], 3: [4.0], 4: [4.0], 5: [6.0]}':4.0,
                    '{0: [4.0], 1: [6.0], 2: [6.0], 3: [4.0], 4: [4.0], 5: [6.0]}':4.0}
                    #'{0: [5.0], 1: [6.0], 2: [6.0], 3: [5.0], 4: [4.0], 5: [6.0]}': 6.0,}
                    #'{0: [5.0], 1: [5.0], 2: [6.0], 3: [5.0], 4: [5.0], 5: [6.0]}': 6.0,
                    #'{0: [6.0], 1: [6.0], 2: [6.0], 3: [6.0], 4: [4.0], 5: [6.0]}':6.0}

        #weakAnswers_dict = {id : res_list[str(responses[id])] for id in self.m_train if str(responses[id]) in res_list and corpusObj.m_label[id] != 5.0}
        weakAnswers_dict = {id: res_list[str(responses[id])] for id in self.m_train if
                            str(responses[id]) in res_list}
        for id in self.m_train:
            if str(responses[id]) == '{0: [5.0], 1: [6.0], 2: [6.0], 3: [5.0], 4: [4.0], 5: [6.0]}':
                weakAnswers_dict[id] = corpusObj.m_label[id]

        coc = 0
        for id in weakAnswers_dict:
            if weakAnswers_dict[id] == corpusObj.m_label[id]:
                coc += 1
        print("correct:", coc, len(weakAnswers_dict), float(coc)/len(weakAnswers_dict))
        '''
        '''
        responses = {}

        for id in range(0, 100):
            res = {0: [4.0], 1: [4.0], 2: [6.0], 3: [4.0], 4: [4.0], 5: [6.0]}
            responses[id] = res
        for id in range(100, 150):
            res = {0: [4.0], 1: [4.0], 2: [4.0], 3: [4.0], 4: [4.0], 5: [6.0]}
            responses[id] = res
        fix_answers = {id:6.0 for id in range(10)}
        '''
        responses = {}
        for id in range(0, 100):
            res = {0: [4.0], 1: [6.0], 2: [6.0], 3: [4.0], 4: [4.0], 5: [6.0]}
            responses[id] = res
        for id in range(100, 101):
            res = {0: [4.0], 1: [4.0], 2: [4.0], 3: [4.0], 4: [4.0], 5: [6.0]}
            responses[id] = res


        classes_list = set(corpusObj.m_label2Idx.keys())
        self.m_prior_pred, self.m_weakOracleCM_pred, ans_distri = DSEM.run(responses,{0:4.0},init_classes=classes_list)
        acc_list = self.computeAccWithCM(self.m_weakOracleCM_pred, self.m_prior_pred)


        correct_num = 0
        total = 0
        id_confid = {}
        for i in range(len(ans_distri)):
            idx = i
            gt_label = 4.0

            ans_i = list(ans_distri[i])
            agg_ans = corpusObj.m_idx2Label[ans_i.index(max(ans_i))]

            correct = gt_label == agg_ans


            if correct:
                correct_num += 1
            total += 1
        print(correct_num, total)

        responses = self.gatherAnsFromWeakOracles(self.m_train, [])
        sampledID = [id for id in self.m_train if corpusObj.m_label[id] == 6.0 and (str(responses[id]) ==
                     '{0: [5.0], 1: [5.0], 2: [6.0], 3: [5.0], 4: [6.0], 5: [6.0]}' or str(responses[id]) ==
                     '{0: [4.0], 1: [6.0], 2: [6.0], 3: [4.0], 4: [4.0], 5: [6.0]}')]
        sampledID.sort()

        sampledID_1 = random.sample(sampledID, int(0.5*(len(sampledID))))
        fix_answers = {id: 4.0 for id in sampledID_1}
        classes_list = set(corpusObj.m_label2Idx.keys())

        responses = self.gatherAnsFromWeakOracles(sampledID, [])
        #correct_num, total, cleanIDTrain, cleanLabelTrain, id_confid, responses = self.generateCleanDataByCrowd_EM(
        #    corpusObj, sampledID, fix_answers)
        self.m_prior_pred, self.m_weakOracleCM_pred, ans_distri = DSEM.run_initT(responses, fix_answers,
                                                                           init_classes=classes_list)
        correct_num = 0
        total =0
        id_confid = {}
        for i in range(len(ans_distri)):
            idx = sampledID[i]
            gt_label = corpusObj.m_label[idx]

            ans_i = list(ans_distri[i])
            confidence_curr = max(ans_i) / sum(ans_i)
            id_confid[idx] = confidence_curr
            agg_ans = corpusObj.m_idx2Label[ans_i.index(max(ans_i))]

            correct = gt_label == agg_ans

            if confidence_curr >= 0:
                if correct:
                    correct_num += 1
                total += 1
        print(correct_num, total)

    def activeTrainClf_cluster_old(self, corpusObj):

        trainSampleNum = []
        # weak oracle answers
        self.m_train.sort()

        # weak global
        '''
        correct_num, total, cleanIDTrain_global, cleanLabelTrain_global, id_confid, responses = self.generateCleanDataByCrowd_EM(corpusObj, self.m_train, dict())
        correct_global, total_global, cleanIDTrain_global, cleanLabelTrain_global, id_conf_global = self.generateCleanDataByCrowd2(self.m_train)
        weakAnswers_dict_global = {cleanIDTrain_global[i]: cleanLabelTrain_global[i] for i in range(len(cleanIDTrain_global))}
        print(correct_num, total)
        correct_global, total_global, cleanIDTrain_global, cleanLabelTrain_global, id_conf_global = self.generateCleanDataByCrowd2(self.m_train)
        print(correct_global,total_global)
        weakAnswers_dict_global_2 = {cleanIDTrain_global[i]: cleanLabelTrain_global[i] for i in
                                   range(len(cleanIDTrain_global))}
        '''

        responses = self.gatherAnsFromWeakOracles(self.m_train, [])


        # weak with cluster
        # clustering: the cluster here is used by the cluster-based crowdsourcing
        #feature0 = corpusObj.m_label[self.m_train]

        feature1 = corpusObj.m_feature[self.m_train]
        feature2 = np.array([[responses[id][orc][0] for orc in responses[id]] for id in self.m_train])  # response
        # one-hot
        onehot_feature = []
        for i in range(len(feature2)):  # each instance
            onehot_feature_i = []
            for j in range(len(feature2[i])):  # eache oracle
                onehotlb = [0 for o in range(len(corpusObj.m_label2Idx))]
                res = feature2[i][j]
                onehotlb[corpusObj.m_label2Idx[res]] = 1
                onehot_feature_i.extend(onehotlb)
            onehot_feature.append(onehot_feature_i)
        feature2 = np.array(onehot_feature) * 100
        #
        # feature1 = np.append(feature1, feature2, axis=1)

        ex_id_cs1, id2cluster1 = self.clusterWithFeatures(26, self.m_train, feature1)

        '''
        # separate the instances in a same cluster but have different labels
        ex_id_cs_new = dd(list)
        clusters = []
        for cid in ex_id_cs1:
            lbs_c = np.unique(corpusObj.m_label[ex_id_cs1[cid]])
            lb2id_c = {lb: [] for lb in lbs_c}
            for id in ex_id_cs1[cid]:
                lb = corpusObj.m_label[id]
                lb2id_c[lb].append(id)
            for lb in lb2id_c:
                clusters.append(lb2id_c[lb])
        for i, ids_c in enumerate(clusters):
            ex_id_cs_new[i] = ids_c
        id2cluster_new = {id: cid for cid in ex_id_cs_new for id in ex_id_cs_new[cid]}
        '''
        '''
        # cluster by class
        ex_id_cs_cls = dd(list)
        id2cluster_cls = {}

        for id in self.m_train:
            lb = corpusObj.m_label[id]
            cid = corpusObj.m_label2Idx[lb]

            ex_id_cs_cls[cid].append(id)
            id2cluster_cls[id] = cid
        '''
        # statistics
        self.ex_id_cs = ex_id_cs1
        self.id2cluster = id2cluster1

        print("cluster num", len(self.ex_id_cs))

        error_lb_dist = []

        # label distribution
        '''
        label_all = corpusObj.m_label[self.m_train]
        unique_lbc_all, counts_lbc_all = np.unique(label_all, return_counts=True)
        dict_lb = {unique_lbc_all[i]: counts_lbc_all[i] for i in range(len(unique_lbc_all))}

        # weak oracle accuracy on each class
        label_oracleAcc = {}  # label: oracle: acc
        for id in self.m_train:
            lb = corpusObj.m_label[id]
            if lb not in label_oracleAcc:
                label_oracleAcc[lb] = {i: 0 for i in range(6)}  # need revise!!!
            for oi in responses[id]:
                response_i = responses[id][oi]
                if response_i[0] == lb:
                    label_oracleAcc[lb][oi] += 1

        print(dict_lb)
        print(label_oracleAcc)
        '''
        '''
        #label distribution per cluster
        label_cluster = {}
        for cid in self.ex_id_cs:
            label_c = corpusObj.m_label[self.ex_id_cs[cid]]
            unique_lbc, counts_lbc = np.unique(label_c, return_counts=True)
            dict_c = {unique_lbc[i]: counts_lbc[i] for i in range(len(unique_lbc))}
            label_cluster[cid] = dict_c
        print(dict_lb)
        print(label_cluster)
        '''

        #c_res_num = self.statics_clusterRes(self.ex_id_cs, responses)

        '''
        c_res_ids_all = {}  # res: ids
        for cid in self.ex_id_cs:
            c_res_ids = {}
            for id in self.ex_id_cs[cid]:
                res_i = str(responses[id])
                if res_i not in c_res_ids:
                    c_res_ids[res_i] = [id]
                else:
                    c_res_ids[res_i].append(id)
            # c_res_ids = sorted(c_res_ids.items(), key=lambda kv: kv[1])
            c_res_ids_all[cid] = c_res_ids
        '''


        # crowdsourcing by cluster

        em_agg_list = []
        em_vt_agg_list = []
        confid_thresh = 0.9
        min_wt = 0.25

        fix_answers = {}

        # EM+WV
        ans_pred_cluster, correct_em, total_em = self.aggregateWeakAns_cluster(responses, fix_answers, corpusObj)
        correct, total, cleanIDTrain, cleanLabelTrain, id_conf = self.generateCleanDataByCrowd_cluster(self.m_train, confid_thresh, min_wt)
        weakAnswers_dict = {cleanIDTrain[i]: cleanLabelTrain[i] for i in range(len(cleanIDTrain))}
        #weakAnswers_dict = {cleanIDTrain[i]: corpusObj.m_label[cleanIDTrain[i]] for i in range(len(cleanIDTrain))}

        agg_acc_list = []
        agg_acc_list.append(float(correct) / total)
        #print(0, float(correct) / total, correct, total)
        #print("em",0, correct_em, total_em)
        #print("wv",0, correct, total)
        em_agg_list.append((0, correct_em, total_em, float(correct_em) / total_em))
        em_vt_agg_list.append((0, correct, total, float(correct) / total))

        # error distribution
        error_lb = np.array([corpusObj.m_label[id] for id in weakAnswers_dict if weakAnswers_dict[id] != corpusObj.m_label[id]])
        values, counts = np.unique(error_lb, return_counts=True)
        error_lb_d = dict(zip(values, counts))
        error_lb_dist.append((0, error_lb_d))

        #weakAnswers_dict = {}
        # clustering
        '''
        # copy crowdsourcing cluster
        self.ex_id = self.ex_id_cs.copy()
        ex_N = [[cid, len(self.ex_id[cid])] for cid in self.ex_id]
        ex = dd(list)
        for cid in self.ex_id:
            c = KMeans(init='k-means++', n_clusters=1, n_init = 10)
            c.fit(corpusObj.m_feature[self.ex_id[cid]])
            dist = np.sort(c.transform(corpusObj.m_feature[self.ex_id[cid]]))  # size: instanceNum x clusterNum
            for id, dis in zip(self.ex_id[cid], dist):
                ex[cid].append([id, dis[0]])

        '''
        c = KMeans(init='k-means++', n_clusters=26, n_init=10)
        c.fit(corpusObj.m_feature[self.m_train])
        dist = np.sort(c.transform(corpusObj.m_feature[self.m_train]))  # size: instanceNum x clusterNum

        ex = dd(list)  # example id, distance to centroid # {clusterID: [[idx, dist],...]}
        self.ex_id = dd(list)  # example id for each C  # {clusterID: [idxs]}
        ex_N = []  # num of examples in each C  #[[clusterID, size]]
        for i, j, k in zip(c.labels_, self.m_train, dist):
            ex[i].append([j, k[0]])
            self.ex_id[i].append(int(j))
        for i, j in ex.items():
            ex[i] = sorted(j, key=lambda x: x[-1])
            ex_N.append([i, len(ex[i])])
        ex_N = sorted(ex_N, key=lambda x: x[-1], reverse=True)

        km_idx = []
        p_idx = []
        p_label = []
        p_dist = dd()

        select_ins_info = []  # (iter, label, response, p_num)

        init_lbs = []

        clusterNum_al = [len(self.ex_id)]

        # first batch of exs: pick centroid of each cluster, and cluster visited based on its size
        ctr = 0
        for ee in ex_N:

            c_idx = ee[0]  # cluster id
            idx = ex[c_idx][0][0]  # id of ex closest to centroid of cluster
            km_idx.append(idx)
            ctr += 1

            lb_idx = corpusObj.m_label[idx]
            init_lbs.append(lb_idx)

            if ctr < 3:  # \ get at least instances for initialization
                continue

            self.update_tao(km_idx, corpusObj)

            p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist, corpusObj)

            # weak

            if ctr % 20 == 0:
                # fix answers
                
                fix_answers = self.getfixAns(km_idx, p_idx, p_label)
                ans_pred_cluster, correct_em, total_em = self.aggregateWeakAns_cluster(responses, fix_answers,
                                                                                       corpusObj)
                correct, total, cleanIDTrain, cleanLabelTrain, id_conf = self.generateCleanDataByCrowd_cluster(
                    self.m_train, confid_thresh, min_wt)

                weakAnswers_dict = {cleanIDTrain[i]: cleanLabelTrain[i] for i in range(len(cleanIDTrain))}
                #weakAnswers_dict = {cleanIDTrain[i]: corpusObj.m_label[cleanIDTrain[i]] for i in
                #                    range(len(cleanIDTrain))}

                error_lb = np.array(
                    [corpusObj.m_label[id] for id in weakAnswers_dict if weakAnswers_dict[id] != corpusObj.m_label[id]])
                values, counts = np.unique(error_lb, return_counts=True)
                error_lb_d = dict(zip(values, counts))
                error_lb_dist.append((ctr, error_lb_d))

                # print("em",ctr, correct_em, total_em)
                # print("wv",ctr, correct, total)
                em_agg_list.append((ctr, correct_em, total_em, float(correct_em) / total_em))
                em_vt_agg_list.append((ctr, correct, total, float(correct) / total))
                #print("em", em_agg_list)
                #print("vt         ", em_vt_agg_list)


            trainNum = 0
            train_correct = 0
            trainNum_al = 0
            train_correct_al = 0
            try:
                acc, trainNum, train_correct = self.get_pred_acc(corpusObj.m_feature[self.m_test],
                                                                 corpusObj.m_label[self.m_test], km_idx, p_idx,
                                                                 p_label, weakAnswers_dict, corpusObj)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print exc_type, e.args, fname, 'on line ' + str(exc_tb.tb_lineno)
                acc = np.nan

            self.m_accList.append(acc)

            # acc of clf trained on only on al
            try:
                acc_al, trainNum_al, train_correct_al = self.get_pred_acc(corpusObj.m_feature[self.m_test],
                                                                          corpusObj.m_label[self.m_test], km_idx, p_idx,
                                                                          p_label, dict(), corpusObj, self.clf_al)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print exc_type, e.args, fname, 'on line ' + str(exc_tb.tb_lineno)
                acc_al = np.nan
            self.m_accList_al.append(acc_al)
            trainSampleNum.append((ctr, trainNum-trainNum_al, train_correct-train_correct_al, trainNum_al, train_correct_al))

        select_ins_info.append((0, init_lbs))

        # fix answers
        fix_answers = self.getfixAns(km_idx, p_idx, p_label)
        ans_pred_cluster, correct_em, total_em = self.aggregateWeakAns_cluster(responses, fix_answers, corpusObj)
        correct, total, cleanIDTrain, cleanLabelTrain, id_conf = self.generateCleanDataByCrowd_cluster(self.m_train, confid_thresh, min_wt)
        #print("em",ctr, correct_em, total_em)
        #print("wv",ctr, correct, total)
        em_agg_list.append((ctr, correct_em, total_em, float(correct_em) / total_em))
        em_vt_agg_list.append((ctr, correct, total, float(correct) / total))
        print("em", em_agg_list)
        print("vt         ", em_vt_agg_list)

        weakAnswers_dict = {cleanIDTrain[i]: cleanLabelTrain[i] for i in range(len(cleanIDTrain))}
        #weakAnswers_dict = {cleanIDTrain[i]: corpusObj.m_label[cleanIDTrain[i]] for i in range(len(cleanIDTrain))}

        # error distribution
        error_lb = np.array(
            [corpusObj.m_label[id] for id in weakAnswers_dict if weakAnswers_dict[id] != corpusObj.m_label[id]])
        values, counts = np.unique(error_lb, return_counts=True)
        error_lb_d = dict(zip(values, counts))
        error_lb_dist.append((ctr, error_lb_d))


        cl_id = []  # track cluster id on each iter
        ex_al = []  # track ex added on each iter
        fn_test = corpusObj.m_feature[self.m_test]
        label_test = corpusObj.m_label[self.m_test]
        for rr in range(ctr, self.m_strongLabelNumThresh):
            '''
            if not p_idx:
                fn_train = corpusObj.m_feature[km_idx]
                label_train = corpusObj.m_label[km_idx]
            else:
                fn_train = corpusObj.m_feature[np.hstack((km_idx, p_idx))]
                label_train = np.hstack((corpusObj.m_label[km_idx], p_label))
            '''
            fn_train, label_train = self.addWeakAns(km_idx, p_idx, p_label, weakAnswers_dict, corpusObj)
            self.clf.fit(fn_train, label_train)
            # acc of clf trained on only on al
            fn_train_al, label_train_al = self.addWeakAns(km_idx, p_idx, p_label, dict(), corpusObj)
            self.clf_al.fit(fn_train_al, label_train_al)

            #if rr < 60:
            #idx, c_idx, = self.select_example_cluster(km_idx, corpusObj, self.clf_al)
            #else:
            idx, c_idx = self.select_example_weak(km_idx, corpusObj, weakAnswers_dict)

            km_idx.append(idx)
            cl_id.append(c_idx)  # track picked cluster id on each iteration
            # ex_al.append([rr,key,v[0][-2],corpusObj.m_label[idx],raw_pt[idx]]) #for debugging

            self.update_tao(km_idx, corpusObj)
            p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist, corpusObj)

            # sub-cluster the cluster
            self.sub_cluster(c_idx, self.clf_al)

            #

            if rr % 20 == 0:
                # update clustering
                #if rr>=60:
                self.revise_cluster(km_idx, p_idx, p_label)
                #self.revise_cluster_AL(km_idx, p_idx, p_label)

                clusterNum_al.append(len(self.ex_id))
                print("cluster al num ", clusterNum_al)

                #self.ex_id_cs = self.ex_id
                #self.id2cluster = {id: cid for cid in self.ex_id for id in self.ex_id[cid]}

                # update agg
                fix_answers = self.getfixAns(km_idx, p_idx, p_label)
                ans_pred_cluster, correct_em, total_em = self.aggregateWeakAns_cluster(responses, fix_answers,
                                                                                       corpusObj)
                correct, total, cleanIDTrain, cleanLabelTrain, id_conf = self.generateCleanDataByCrowd_cluster(
                    self.m_train, confid_thresh, min_wt)

                weakAnswers_dict = {cleanIDTrain[i]: cleanLabelTrain[i] for i in range(len(cleanIDTrain))}
                #weakAnswers_dict = {cleanIDTrain[i]: corpusObj.m_label[cleanIDTrain[i]] for i in
                #                    range(len(cleanIDTrain))}

                # print("em",ctr, correct_em, total_em)
                # print("wv",ctr, correct, total)
                em_agg_list.append((rr, correct_em, total_em, float(correct_em) / total_em))
                em_vt_agg_list.append((rr, correct, total, float(correct) / total))

                # error distribution
                error_lb = np.array(
                    [corpusObj.m_label[id] for id in weakAnswers_dict if weakAnswers_dict[id] != corpusObj.m_label[id]])
                values, counts = np.unique(error_lb, return_counts=True)
                error_lb_d = dict(zip(values, counts))
                error_lb_dist.append((rr, error_lb_d))

            acc, trainNum, train_correct = self.get_pred_acc(fn_test, label_test, km_idx, p_idx, p_label,
                                                             weakAnswers_dict, corpusObj)
            self.m_accList.append(acc)

            # acc of clf trained on only on al
            acc_al, trainNum_al, train_correct_al = self.get_pred_acc(corpusObj.m_feature[self.m_test],
                                                                      corpusObj.m_label[self.m_test], km_idx, p_idx,
                                                                      p_label, dict(), corpusObj, self.clf_al)

            self.m_accList_al.append(acc_al)
            trainSampleNum.append((rr, trainNum - trainNum_al, train_correct -train_correct_al, trainNum_al, train_correct_al))
            select_ins_info.append((rr, corpusObj.m_label[idx], responses[idx], trainNum - trainNum_al))
            if rr % 20 == 0:
                # print(rr, acc)
                # print("RMSE: ", str(rr), str(RMSE_list))
                # print("agg acc: ",str(rr), str(agg_acc_list), str(correct), str(total))
                acc_diff = [(i, self.m_accList[i] - self.m_accList_al[i]) for i in range(len(self.m_accList))]
                print(rr, acc_diff)
                print(rr, trainSampleNum)
                print(rr, error_lb_dist)
                # print(select_ins_info)

        print("finished!")

    def activeTrainClf_cluster_old_orgin(self, corpusObj):

        trainSampleNum = []
        # weak oracle answers
        self.m_train.sort()

        # weak global
        '''
        correct_num, total, cleanIDTrain_global, cleanLabelTrain_global, id_confid, responses = self.generateCleanDataByCrowd_EM(corpusObj, self.m_train, dict())
        #correct_global, total_global, cleanIDTrain_global, cleanLabelTrain_global, id_conf_global = self.generateCleanDataByCrowd2(self.m_train)
        weakAnswers_dict_global = {cleanIDTrain_global[i]: cleanLabelTrain_global[i] for i in range(len(cleanIDTrain_global))}
        print(correct_num, total)
        correct_global, total_global, cleanIDTrain_global, cleanLabelTrain_global, id_conf_global = self.generateCleanDataByCrowd2(self.m_train)
        weakAnswers_dict_global_2 = {cleanIDTrain_global[i]: cleanLabelTrain_global[i] for i in
                                   range(len(cleanIDTrain_global))}
        '''

        responses = self.gatherAnsFromWeakOracles(self.m_train, [])

        # weak with cluster
        # clustering: the cluster here is used by the cluster-based crowdsourcing
        feature1 = corpusObj.m_feature[self.m_train]
        feature2 = np.array([[responses[id][orc][0] for orc in responses[id]] for id in self.m_train])  # response
        # one-hot
        onehot_feature = []
        for i in range(len(feature2)):  # each instance
            onehot_feature_i = []
            for j in range(len(feature2[i])):  # eache oracle
                onehotlb = [0 for o in range(len(corpusObj.m_label2Idx))]
                res = feature2[i][j]
                onehotlb[corpusObj.m_label2Idx[res]] = 1
                onehot_feature_i.extend(onehotlb)
            onehot_feature.append(onehot_feature_i)
        feature2 = np.array(onehot_feature) * 100
        #
        #feature1 = np.append(feature1, feature2, axis=1)

        ex_id_cs1, id2cluster1 = self.clusterWithFeatures(4, self.m_train, feature1)

        # statistics
        self.ex_id_cs = ex_id_cs1
        self.id2cluster = id2cluster1


        # label distribution
        '''
        label_all = corpusObj.m_label[self.m_train]
        unique_lbc_all, counts_lbc_all = np.unique(label_all, return_counts=True)
        dict_lb = {unique_lbc_all[i]: counts_lbc_all[i] for i in range(len(unique_lbc_all))}

        # weak oracle accuracy on each class
        label_oracleAcc = {}  # label: oracle: acc
        for id in self.m_train:
            lb = corpusObj.m_label[id]
            if lb not in label_oracleAcc:
                label_oracleAcc[lb] = {i: 0 for i in range(6)}  # need revise!!!
            for oi in responses[id]:
                response_i = responses[id][oi]
                if response_i[0] == lb:
                    label_oracleAcc[lb][oi] += 1

        print(dict_lb)
        print(label_oracleAcc)
        '''
        '''
        #label distribution per cluster
        label_cluster = {}
        for cid in self.ex_id_cs:
            label_c = corpusObj.m_label[self.ex_id_cs[cid]]
            unique_lbc, counts_lbc = np.unique(label_c, return_counts=True)
            dict_c = {unique_lbc[i]: counts_lbc[i] for i in range(len(unique_lbc))}
            label_cluster[cid] = dict_c
        print(dict_lb)
        print(label_cluster)
        '''

        c_res_num = self.statics_clusterRes(self.ex_id_cs, responses)

        '''
        c_res_ids_all = {}  # res: ids
        for cid in self.ex_id_cs:
            c_res_ids = {}
            for id in self.ex_id_cs[cid]:
                res_i = str(responses[id])
                if res_i not in c_res_ids:
                    c_res_ids[res_i] = [id]
                else:
                    c_res_ids[res_i].append(id)
            # c_res_ids = sorted(c_res_ids.items(), key=lambda kv: kv[1])
            c_res_ids_all[cid] = c_res_ids
        '''

        # crowdsourcing by cluster
        fix_answers = {}
        # feature 1
        _, _, _ = self.aggregateWeakAns_cluster(responses, fix_answers, corpusObj)
        correct, total, cleanIDTrain, cleanLabelTrain, id_conf = self.generateCleanDataByCrowd_cluster(self.m_train)
        weakAnswers_dict = {cleanIDTrain[i]: cleanLabelTrain[i] for i in range(len(cleanIDTrain))}

        agg_acc_list = []
        agg_acc_list.append(float(correct) / total)
        print(0, float(correct) / total, correct, total)

        '''
        # filter
        rate = 0.25
        filtered_id = self.filter_from_cluster(self.m_train, self.ex_id_cs, self.id2cluster, responses, c_res_num, rate)
        #weakAnswers_dict = {ft_id: weakAnswers_dict[ft_id] for ft_id in filtered_id}
        weakAnswers_dict = {ft_id: weakAnswers_dict[ft_id] for ft_id in filtered_id if id_conf[ft_id] >=0.8}
        #weakAnswers_dict = {ft_id: weakAnswers_dict[ft_id] for ft_id in filtered_id if weakAnswers_dict[ft_id] == corpusObj.m_label[ft_id]}
        #weakAnswers_dict = {ft_id: weakAnswers_dict[ft_id] for ft_id in weakAnswers_dict if id_conf[ft_id] >=0.8}

        weak_label_dist = {}
        weak_label_id = {}
        weak_label_correct = {}

        for id in weakAnswers_dict:
            ans = weakAnswers_dict[id]
            cr = corpusObj.m_label[id] == weakAnswers_dict[id]
            if ans not in weak_label_dist:
                weak_label_dist[ans] = 1
                weak_label_id[ans] = [id]
                if cr:
                    weak_label_correct[ans] = 1
                else:
                    weak_label_correct[ans] = 0
            else:
                weak_label_dist[ans] += 1
                weak_label_id[ans].append(id)
                if cr:
                    weak_label_correct[ans] += 1
        print("weak label distribution: ", weak_label_dist, weak_label_correct)
        '''
        # balance
        '''
        min_labelNum = 99999
        for lb in weak_label_dist:
            if weak_label_dist[lb] < min_labelNum:
                min_labelNum = weak_label_dist[lb]

        filtered_id = {}
        for lb in weak_label_id:
            weak_label_id[lb] = random.sample(weak_label_id[lb], min_labelNum)
            for id in weak_label_id[lb]:
                filtered_id[id] = 1

        weakAnswers_dict = {ft_id: weakAnswers_dict[ft_id] for ft_id in filtered_id}

        weak_label_dist = {}
        weak_label_id = {}
        weak_label_correct = {}

        for id in weakAnswers_dict:
            ans = weakAnswers_dict[id]
            cr = corpusObj.m_label[id] == weakAnswers_dict[id]
            if ans not in weak_label_dist:
                weak_label_dist[ans] = 1
                weak_label_id[ans] = [id]
                if cr:
                    weak_label_correct[ans] = 1
                else:
                    weak_label_correct[ans] = 0
            else:
                weak_label_dist[ans] += 1
                weak_label_id[ans].append(id)
                if cr:
                    weak_label_correct[ans] += 1
        print("weak label distribution: ", weak_label_dist, weak_label_correct)
        # balance>
        '''

        '''
        correct_ft = 0
        for id in weakAnswers_dict:
            if weakAnswers_dict[id] == corpusObj.m_label[id]:
                correct_ft += 1
        print("filtered acc", correct_ft, len(weakAnswers_dict), float(correct_ft)/len(weakAnswers_dict))
        '''
        #acc, trainNum, train_correct = self.get_pred_acc(corpusObj.m_feature[self.m_test],
        #                                                 corpusObj.m_label[self.m_test], [], [],
        #                                                 [], weakAnswers_dict, corpusObj)
        #print(acc)

        '''
        # true acc per cluster for each oracle
        acc_cluster_true = {}
        if self.m_useTransfer:
            for i in range(len(corpusObj.m_transferLabelList)):
                acc_cluster_true_i = {}
                for cid in self.ex_id_cs:
                    ex_ids_c = self.ex_id_cs[cid]
                    acc_cluster_true_i[cid] = accuracy_score(corpusObj.m_label[ex_ids_c], corpusObj.m_transferLabelList[i][ex_ids_c])
                acc_cluster_true[i] = dict(acc_cluster_true_i)

        RMSE_list = []
        RMSE = 0.0
        for i in range(len(corpusObj.m_transferLabelList)):
            for cid in self.ex_id_cs:
                RMSE += math.pow(acc_cluster_true[i][cid] - self.m_accTransfer_cluster_pred[cid][i], 2)
        RMSE = math.sqrt(RMSE / (len(corpusObj.m_transferLabelList)*len(self.ex_id_cs)))
        RMSE_list.append(RMSE)

        #print(0, RMSE_list)
        #print(0, agg_acc_list)
        '''


        # clustering
        c = KMeans(init='k-means++', n_clusters=26, n_init=10)
        c.fit(corpusObj.m_feature[self.m_train])
        dist = np.sort(c.transform(corpusObj.m_feature[self.m_train]))  # size: instanceNum x clusterNum

        ex = dd(list)  # example id, distance to centroid # {clusterID: [[idx, dist],...]}
        self.ex_id = dd(list)  # example id for each C  # {clusterID: [idxs]}
        ex_N = []  # num of examples in each C  #[[clusterID, size]]
        for i, j, k in zip(c.labels_, self.m_train, dist):
            ex[i].append([j, k[0]])
            self.ex_id[i].append(int(j))
        for i, j in ex.items():
            ex[i] = sorted(j, key=lambda x: x[-1])
            ex_N.append([i, len(ex[i])])
        ex_N = sorted(ex_N, key=lambda x: x[-1], reverse=True)

        km_idx = []
        p_idx = []
        p_label = []
        p_dist = dd()

        select_ins_info = [] #(iter, label, response, p_num)

        init_lbs = []
        # first batch of exs: pick centroid of each cluster, and cluster visited based on its size
        ctr = 0
        for ee in ex_N:

            c_idx = ee[0]  # cluster id
            idx = ex[c_idx][0][0]  # id of ex closest to centroid of cluster
            km_idx.append(idx)
            ctr += 1

            lb_idx = corpusObj.m_label[idx]
            init_lbs.append(lb_idx)

            if ctr < 3:  # \ get at least instances for initialization
                continue

            self.update_tao(km_idx, corpusObj)

            p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist, corpusObj)

            # weak
            '''
            if ctr % 20 == 0:
                
                fix_answers = self.getfixAns(km_idx, p_idx, p_label)
                _, _, _ = self.aggregateWeakAns_cluster(responses, fix_answers, corpusObj)
                correct, total, cleanIDTrain, cleanLabelTrain, id_conf = self.generateCleanDataByCrowd_cluster(self.m_train)
                weakAnswers_dict = {cleanIDTrain[i]: cleanLabelTrain[i] for i in range(len(cleanIDTrain))}

                #filter
                
                weakAnswers_dict = {ft_id: weakAnswers_dict[ft_id] for ft_id in filtered_id}

                correct_ft = 0
                for id in weakAnswers_dict:
                    if weakAnswers_dict[id] == corpusObj.m_label[id]:
                        correct_ft += 1
                print("filtered acc", correct_ft, len(weakAnswers_dict))
                

                #agg_acc_list.append(float(correct) / total)
                agg_acc_list.append(float(correct_ft) / len(weakAnswers_dict))
                RMSE = 0.0
                for i in range(len(corpusObj.m_transferLabelList)):
                    for cid in self.ex_id_cs:
                        RMSE += math.pow(acc_cluster_true[i][cid] - self.m_accTransfer_cluster_pred[cid][i], 2)
                RMSE = math.sqrt(RMSE / (len(corpusObj.m_transferLabelList) * len(self.ex_id_cs)))
                RMSE_list.append(RMSE)
            '''
            trainNum = 0
            train_correct = 0
            trainNum_al = 0
            train_correct_al = 0
            try:
                acc, trainNum, train_correct = self.get_pred_acc(corpusObj.m_feature[self.m_test], corpusObj.m_label[self.m_test], km_idx, p_idx,
                                                                 p_label, weakAnswers_dict, corpusObj)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print exc_type, e.args, fname, 'on line ' + str(exc_tb.tb_lineno)
                acc = np.nan

            self.m_accList.append(acc)

            # acc of clf trained on only on al
            try:
                acc_al, trainNum_al, train_correct_al = self.get_pred_acc(corpusObj.m_feature[self.m_test], corpusObj.m_label[self.m_test], km_idx, p_idx,
                                        p_label, dict(), corpusObj, self.clf_al)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print exc_type, e.args, fname, 'on line ' + str(exc_tb.tb_lineno)
                acc_al = np.nan
            self.m_accList_al.append(acc_al)
            trainSampleNum.append((trainNum, train_correct, trainNum_al, train_correct_al))

        select_ins_info.append((0, init_lbs))

        cl_id = []  # track cluster id on each iter
        ex_al = []  # track ex added on each iter
        fn_test = corpusObj.m_feature[self.m_test]
        label_test = corpusObj.m_label[self.m_test]
        for rr in range(ctr, self.m_strongLabelNumThresh):
            '''
            if not p_idx:
                fn_train = corpusObj.m_feature[km_idx]
                label_train = corpusObj.m_label[km_idx]
            else:
                fn_train = corpusObj.m_feature[np.hstack((km_idx, p_idx))]
                label_train = np.hstack((corpusObj.m_label[km_idx], p_label))
            '''
            fn_train, label_train = self.addWeakAns(km_idx, p_idx, p_label, weakAnswers_dict, corpusObj)
            self.clf.fit(fn_train, label_train)
            # acc of clf trained on only on al
            fn_train_al, label_train_al = self.addWeakAns(km_idx, p_idx, p_label, dict(), corpusObj)
            self.clf_al.fit(fn_train_al, label_train_al)


            idx, c_idx, = self.select_example_cluster(km_idx, corpusObj, self.clf_al)

            km_idx.append(idx)
            cl_id.append(c_idx)  # track picked cluster id on each iteration
            # ex_al.append([rr,key,v[0][-2],corpusObj.m_label[idx],raw_pt[idx]]) #for debugging

            self.update_tao(km_idx, corpusObj)
            p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist, corpusObj)

            #
            '''
            if rr % 20 == 0:
                
                fix_answers = self.getfixAns(km_idx, p_idx, p_label)
                _, _, _ = self.aggregateWeakAns_cluster(responses, fix_answers, corpusObj)
                correct, total, cleanIDTrain, cleanLabelTrain, id_conf = self.generateCleanDataByCrowd_cluster(self.m_train)
                weakAnswers_dict = {cleanIDTrain[i]: cleanLabelTrain[i] for i in range(len(cleanIDTrain))}

                # filter
                weakAnswers_dict = {ft_id: weakAnswers_dict[ft_id] for ft_id in filtered_id}

                correct_ft = 0
                for id in weakAnswers_dict:
                    if weakAnswers_dict[id] == corpusObj.m_label[id]:
                        correct_ft += 1
                print("filtered acc", correct_ft, len(weakAnswers_dict))
                
                if len(weakAnswers_dict) > 0:
                    agg_acc_list.append(float(correct_ft) / len(weakAnswers_dict))
                else:
                    agg_acc_list.append(0.0)

                RMSE = 0.0
                for i in range(len(corpusObj.m_transferLabelList)):
                    for cid in self.ex_id_cs:
                        RMSE += math.pow(acc_cluster_true[i][cid] - self.m_accTransfer_cluster_pred[cid][i], 2)
                RMSE = math.sqrt(RMSE / (len(corpusObj.m_transferLabelList) * len(self.ex_id_cs)))
                RMSE_list.append(RMSE)
            '''
            acc, trainNum, train_correct = self.get_pred_acc(fn_test, label_test, km_idx, p_idx, p_label, weakAnswers_dict, corpusObj)
            self.m_accList.append(acc)

            # acc of clf trained on only on al
            acc_al, trainNum_al, train_correct_al = self.get_pred_acc(corpusObj.m_feature[self.m_test], corpusObj.m_label[self.m_test], km_idx, p_idx,
                                       p_label, dict(), corpusObj, self.clf_al)

            self.m_accList_al.append(acc_al)
            trainSampleNum.append((trainNum, train_correct, trainNum_al, train_correct_al))
            select_ins_info.append((rr, corpusObj.m_label[idx], responses[idx], trainNum-trainNum_al))
            if rr % 20 == 0:
                #print(rr, acc)
                #print("RMSE: ", str(rr), str(RMSE_list))
                #print("agg acc: ",str(rr), str(agg_acc_list), str(correct), str(total))
                acc_diff = [self.m_accList[i]-self.m_accList_al[i] for i in range(len(self.m_accList))]
                print(rr, acc_diff)
                #print(rr, trainSampleNum)
                print(select_ins_info)

        print("finished!")

    def clusterWithFeatures(self, clusterNum, selectedID, features_cluster):
        ex_id_cs = dd(list) # example id for each C  # {clusterID: [idxs]}
        id2cluster = {}
        c_cs = KMeans(init='k-means++', n_clusters=clusterNum, n_init=10)
        c_cs.fit(features_cluster)
        dist_cs = np.sort(c_cs.transform(features_cluster))  # size: instanceNum x clusterNum

        for i, j, k in zip(c_cs.labels_, selectedID, dist_cs):
            ex_id_cs[i].append(int(j))
            id2cluster[j] = i
        return ex_id_cs, id2cluster

    def statics_clusterRes(self, ex_id_cs, response):
        c_res = {} # cid: response: num
        for cid in ex_id_cs:
            c_res_i = {}
            for id in ex_id_cs[cid]:
                res_i = str(response[id])
                if res_i not in c_res_i:
                    c_res_i[res_i] = 1
                else:
                    c_res_i[res_i] += 1
            #c_res_i = sorted(c_res_i.items(), key=lambda kv: kv[1], reverse=True)
            c_res[cid] = c_res_i
        return c_res

    def activeTrainClf_cluster(self, corpusObj):
        # weak oracle answers
        self.m_train.sort()
        '''
        # weak global
        _, _, _, _, _, responses = self.generateCleanDataByCrowd_EM(corpusObj, self.m_train, dict(strongAnswers_dict))
        correct_global, total_global, cleanIDTrain_global, cleanLabelTrain_global, id_conf_global = self.generateCleanDataByCrowd2(self.m_train)

        cleanFeatureTrain = corpusObj.m_feature[cleanIDTrain_global]
        weakAnswers_dict_global = {cleanIDTrain_global[i]: cleanLabelTrain_global[i] for i in range(len(cleanIDTrain_global))}
        '''

        responses = self.gatherAnsFromWeakOracles(self.m_train, [])

        current_truth = {} # current strong answers + pred truth

        # weak with cluster
        # clustering: the cluster here is used by the cluster-based crowdsourcing
        feature1 = corpusObj.m_feature[self.m_train]
        feature2 = np.array([[responses[id][orc][0] for orc in responses[id]] for id in self.m_train]) # response
        # one-hot
        onehot_feature = []
        for i in range(len(feature2)): # each instance
            onehot_feature_i = []
            for j in range(len(feature2[i])): # eache oracle
                onehotlb = [0 for o in range(len(corpusObj.m_label2Idx))]
                res = feature2[i][j]
                onehotlb[corpusObj.m_label2Idx[res]] = 1
                onehot_feature_i.extend(onehotlb)
            onehot_feature.append(onehot_feature_i)
        feature2 = onehot_feature
        #
        feature1 = np.append(feature1, feature2, axis=1)
        #feature2
        #feature3 = np.array([[responses[id][orc][0] == current_truth[id] for orc in responses[id]] for id in self.m_train])# [response == pred true]
        ex_id_cs1, id2cluster1 = self.clusterWithFeatures(4, self.m_train, feature1)
        ex_id_cs2, id2cluster2 = self.clusterWithFeatures(4, self.m_train, feature2)


        # statistics
        self.ex_id_cs = ex_id_cs1
        self.id2cluster = id2cluster1

        c_res_num = self.statics_clusterRes(self.ex_id_cs, responses)

        c_res_ids_all = {} # res: ids
        for cid in self.ex_id_cs:
            c_res_ids = {}
            for id in self.ex_id_cs[cid]:
                res_i = str(responses[id])
                if res_i not in c_res_ids:
                    c_res_ids[res_i] = [id]
                else:
                    c_res_ids[res_i].append(id)
            #c_res_ids = sorted(c_res_ids.items(), key=lambda kv: kv[1])
            c_res_ids_all[cid] = c_res_ids

        # filter
        rate = 0.25
        filtered_id = []
        for id in self.m_train:
            res_i = str(responses[id])
            cid = self.id2cluster[id]
            cid_size = len(self.ex_id_cs[cid])
            if c_res_num[cid][res_i] >= rate * cid_size:
                filtered_id.append(id)


        #statistics
        unique_label, counts_label = np.unique(corpusObj.m_label[self.m_train], return_counts=True)
        label_distri_train = dict(zip(unique_label, counts_label))

        unique_label, counts_label = np.unique(corpusObj.m_label[self.m_test], return_counts=True)
        label_distri_test = dict(zip(unique_label, counts_label))
        print(label_distri_train, label_distri_test)

        compareResTrue = {id: (corpusObj.m_label[id], responses[id]) for id in responses}
        oc = OrderedDict(sorted(compareResTrue.items(), key=lambda t: t[0]))
        oc_list = []
        for key, value in oc.iteritems():
            oc_list.append((key, value))

        oc_list_1 = oc_list[:200]
        oc_list_2 = oc_list[200:]

        current_truth = {} # current strong answers + pred truth

        true_respond = {str(lb): {} for lb in corpusObj.m_label2Idx}
        for id in self.m_train:
            lb = str(corpusObj.m_label[id])
            resp = str(responses[id])
            if resp not in true_respond[lb]:
                true_respond[lb][resp] = 1
            else:
                true_respond[lb][resp] += 1

        respond_true = {}
        for id in self.m_train:
            lb = str(corpusObj.m_label[id])
            resp = str(responses[id])
            if resp not in respond_true:
                respond_true[resp] = {lb: 1}
            else:
                if lb not in respond_true[resp]:
                    respond_true[resp][lb] = 1
                else:
                    respond_true[resp][lb] += 1

        # statistics/>

        self.ex_id_cs = ex_id_cs1
        self.id2cluster = id2cluster1

        # crowdsourcing by cluster
        fix_answers = {}

        # feature 1
        _, _, _ = self.aggregateWeakAns_cluster(responses, fix_answers, corpusObj)
        correct, total, cleanIDTrain, cleanLabelTrain, id_conf = self.generateCleanDataByCrowd_cluster(self.m_train)
        weakAnswers_dict_all = {cleanIDTrain[i]: cleanLabelTrain[i] for i in range(len(cleanIDTrain))}

        confid_thresh = 0.0
        weakAnswers_dict = {cleanIDTrain[i]: cleanLabelTrain[i] for i in range(len(cleanIDTrain)) if id_conf[cleanIDTrain[i]] >= confid_thresh}

        agg_acc_list = []
        agg_acc_list.append(float(correct)/total)

        # filter:
        weakAnswers_dict_filter = {ft_id: weakAnswers_dict_all[ft_id] for ft_id in filtered_id}

        correct_ft = 0
        for id in weakAnswers_dict_filter:
            if weakAnswers_dict_filter[id] == corpusObj.m_label[id]:
                correct_ft += 1

        print("filtered acc",correct_ft,len(weakAnswers_dict_filter))



        '''
        # statistics: true acc per cluster for each oracle
        acc_cluster_true = {}
        if self.m_useTransfer:
            for i in range(len(corpusObj.m_transferLabelList)):
                acc_cluster_true_i = {}
                for cid in self.ex_id_cs:
                    ex_ids_c = self.ex_id_cs[cid]
                    acc_cluster_true_i[cid] = accuracy_score(corpusObj.m_label[ex_ids_c], corpusObj.m_transferLabelList[i][ex_ids_c])
                acc_cluster_true[i] = dict(acc_cluster_true_i)

        RMSE_list = []
        RMSE = 0.0
        for i in range(len(corpusObj.m_transferLabelList)):
            for cid in self.ex_id_cs:
                RMSE += math.pow(acc_cluster_true[i][cid] - self.m_accTransfer_cluster_pred[cid][i], 2)
        RMSE = math.sqrt(RMSE / (len(corpusObj.m_transferLabelList)*len(self.ex_id_cs)))
        RMSE_list.append(RMSE)
        '''

        # feature 2
        self.ex_id_cs = ex_id_cs2
        self.id2cluster = id2cluster2

        _, _, _ = self.aggregateWeakAns_cluster(responses, fix_answers, corpusObj)
        correct_f2, total_f2, cleanIDTrain_f2, cleanLabelTrain_f2, id_conf_f2 = self.generateCleanDataByCrowd_cluster(
            self.m_train)
        weakAnswers_dict_f2 = {cleanIDTrain_f2[i]: cleanLabelTrain_f2[i] for i in range(len(cleanIDTrain_f2))}

        agg_acc_list_f2 = []
        agg_acc_list_f2.append(float(correct_f2)/total_f2)
        # feature 2 />
        '''
        # statistics: true acc per cluster for each oracle
        acc_cluster_true_f2 = {}
        if self.m_useTransfer:
            for i in range(len(corpusObj.m_transferLabelList)):
                acc_cluster_true_i = {}
                for cid in self.ex_id_cs:
                    ex_ids_c = self.ex_id_cs[cid]
                    acc_cluster_true_i[cid] = accuracy_score(corpusObj.m_label[ex_ids_c], corpusObj.m_transferLabelList[i][ex_ids_c])
                acc_cluster_true_f2[i] = dict(acc_cluster_true_i)

        RMSE_list_f2 = []
        RMSE = 0.0
        for i in range(len(corpusObj.m_transferLabelList)):
            for cid in self.ex_id_cs:
                RMSE += math.pow(acc_cluster_true_f2[i][cid] - self.m_accTransfer_cluster_pred[cid][i], 2)
        RMSE = math.sqrt(RMSE / (len(corpusObj.m_transferLabelList)*len(self.ex_id_cs)))
        RMSE_list_f2.append(RMSE)

        print(RMSE_list, RMSE_list_f2)

        # set confid = 0.0 before run this module
        # confidence - accuracy
        cf_thresh = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99995]
        total_cf = [0 for i in range(len(cf_thresh))]
        correct_cf = [0 for i in range(len(cf_thresh))]
        #total_cf_global = [0 for i in range(len(cf_thresh))]
        #correct_cf_global = [0 for i in range(len(cf_thresh))]
        for id in id_conf:
            correct = corpusObj.m_label[id] == weakAnswers_dict[id]
            #correct_global = corpusObj.m_label[id] == weakAnswers_dict_global[id]
            for cf_i in range(len(cf_thresh)):
                cfs = cf_thresh[cf_i]
                #if id_conf_global[id] >= cfs:
                #    total_cf_global[cf_i] += 1
                #    if correct_global:
                #        correct_cf_global[cf_i] += 1
                if id_conf[id] >= cfs:
                    total_cf[cf_i] += 1
                    if correct:
                        correct_cf[cf_i] += 1
        acc_cf = [float(correct_cf[i]) / total_cf[i] for i in range(len(cf_thresh)) if total_cf[i] > 0]
        #acc_cf_global = [float(correct_cf_global[i]) / total_cf_global[i] for i in range(len(cf_thresh)) if total_cf_global[i] > 0]
        #print(correct_cf, total_cf, correct_cf_global, total_cf_global)
        #print(acc_cf, acc_cf_global)
        
        total_cf_f2 = [0 for i in range(len(cf_thresh))]
        correct_cf_f2 = [0 for i in range(len(cf_thresh))]
        for id in id_conf_f2:
            correct_f2 = corpusObj.m_label[id] == weakAnswers_dict_f2[id]
            for cf_i in range(len(cf_thresh)):
                cfs = cf_thresh[cf_i]
                if id_conf_f2[id] >= cfs:
                    total_cf_f2[cf_i] += 1
                    if correct_f2:
                        correct_cf_f2[cf_i] += 1
        acc_cf_f2 = [float(correct_cf_f2[i]) / total_cf_f2[i] for i in range(len(cf_thresh)) if total_cf_f2[i] > 0]
        print(correct_cf, total_cf, correct_cf_f2, total_cf_f2)
        print(acc_cf, acc_cf_f2)
        
        '''



        # clustering
        c = KMeans(init='k-means++', n_clusters=26, n_init=10)
        c.fit(corpusObj.m_feature[self.m_train])
        dist = np.sort(c.transform(corpusObj.m_feature[self.m_train]))  # size: instanceNum x clusterNum

        ex = dd(list)  # example id, distance to centroid # {clusterID: [[idx, dist],...]}
        self.ex_id = dd(list)  # example id for each C  # {clusterID: [idxs]}
        ex_N = []  # num of examples in each C  #[[clusterID, size]]
        for i, j, k in zip(c.labels_, self.m_train, dist):
            ex[i].append([j, k[0]])
            self.ex_id[i].append(int(j))
        for i, j in ex.items():
            ex[i] = sorted(j, key=lambda x: x[-1])
            ex_N.append([i, len(ex[i])])
        ex_N = sorted(ex_N, key=lambda x: x[-1], reverse=True)

        km_idx = []
        p_idx = []
        p_label = []
        p_dist = dd()
        # first batch of exs: pick centroid of each cluster, and cluster visited based on its size
        ctr = 0
        for ee in ex_N:

            c_idx = ee[0]  # cluster id
            idx = ex[c_idx][0][0]  # id of ex closest to centroid of cluster
            km_idx.append(idx)
            ctr += 1

            if ctr < 3:  # \ get at least instances for initialization
                continue

            self.update_tao(km_idx, corpusObj)

            p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist, corpusObj)

            # weak
            if ctr % 20 == 0:
                fix_answers = self.getfixAns(km_idx, p_idx, p_label)
                _, _, _ = self.aggregateWeakAns_cluster(responses, fix_answers, corpusObj)
                correct, total, cleanIDTrain, cleanLabelTrain, id_conf = self.generateCleanDataByCrowd_cluster(self.m_train)
                weakAnswers_dict_all = {cleanIDTrain[i]: cleanLabelTrain[i] for i in range(len(cleanIDTrain))}

                weakAnswers_dict = {cleanIDTrain[i]: cleanLabelTrain[i] for i in range(len(cleanIDTrain)) if
                                    id_conf[cleanIDTrain[i]] >= confid_thresh}

                agg_acc_list.append(float(correct) / total)
                RMSE = 0.0
                for i in range(len(corpusObj.m_transferLabelList)):
                    for cid in self.ex_id_cs:
                        RMSE += math.pow(acc_cluster_true[i][cid] - self.m_accTransfer_cluster_pred[cid][i], 2)
                RMSE = math.sqrt(RMSE / (len(corpusObj.m_transferLabelList) * len(self.ex_id_cs)))
                RMSE_list.append(RMSE)

                #< feature 3
                current_truth = fix_answers
                for wid in weakAnswers_dict:
                    if wid not in fix_answers:
                        current_truth[wid] = weakAnswers_dict_all[wid]
                feature3 = np.array([[responses[id][orc][0] == current_truth[id] for orc in responses[id]] for id in
                                     self.m_train])  # [response == pred true]
                ex_id_cs3, id2cluster3 = self.clusterWithFeatures(3, self.m_train, feature3)

                self.ex_id_cs = ex_id_cs3
                self.id2cluster = id2cluster3

                _, _, _ = self.aggregateWeakAns_cluster(responses, fix_answers, corpusObj)
                correct_f2, total_f2, cleanIDTrain_f2, cleanLabelTrain_f2, id_conf_f2 = self.generateCleanDataByCrowd_cluster(
                    self.m_train)
                weakAnswers_dict_f2 = {cleanIDTrain_f2[i]: cleanLabelTrain_f2[i] for i in range(len(cleanIDTrain_f2))}

                # feature 3>

            try:
                acc = self.get_pred_acc(corpusObj.m_feature[self.m_test], corpusObj.m_label[self.m_test], km_idx, p_idx, p_label, weakAnswers_dict, corpusObj)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print exc_type, e.args, fname, 'on line ' + str(exc_tb.tb_lineno)
                acc = np.nan

            self.m_accList.append(acc)
            # print("acc\t", acc)

        cl_id = []  # track cluster id on each iter
        ex_al = []  # track ex added on each iter
        fn_test = corpusObj.m_feature[self.m_test]
        label_test = corpusObj.m_label[self.m_test]
        for rr in range(ctr, self.m_strongLabelNumThresh):
            '''
            if not p_idx:
                fn_train = corpusObj.m_feature[km_idx]
                label_train = corpusObj.m_label[km_idx]
            else:
                fn_train = corpusObj.m_feature[np.hstack((km_idx, p_idx))]
                label_train = np.hstack((corpusObj.m_label[km_idx], p_label))
            '''
            fn_train, label_train = self.addWeakAns(km_idx, p_idx, p_label, weakAnswers_dict, corpusObj)
            self.clf.fit(fn_train, label_train)

            idx, c_idx, = self.select_example_cluster(km_idx, corpusObj)
            km_idx.append(idx)
            cl_id.append(c_idx)  # track picked cluster id on each iteration
            # ex_al.append([rr,key,v[0][-2],corpusObj.m_label[idx],raw_pt[idx]]) #for debugging

            self.update_tao(km_idx, corpusObj)
            p_idx, p_label, p_dist = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist, corpusObj)

            #

            if rr % 20 == 0:
                fix_answers = self.getfixAns(km_idx, p_idx, p_label)
                _, _, _ = self.aggregateWeakAns_cluster(responses, fix_answers, corpusObj)
                correct, total, cleanIDTrain, cleanLabelTrain, id_conf = self.generateCleanDataByCrowd_cluster(self.m_train)
                weakAnswers_dict = {cleanIDTrain[i]: cleanLabelTrain[i] for i in range(len(cleanIDTrain))}

                if total > 0:
                    agg_acc_list.append(float(correct) / total)
                else:
                    agg_acc_list.append(0.0)

                RMSE = 0.0
                for i in range(len(corpusObj.m_transferLabelList)):
                    for cid in self.ex_id_cs:
                        RMSE += math.pow(acc_cluster_true[i][cid] - self.m_accTransfer_cluster_pred[cid][i], 2)
                RMSE = math.sqrt(RMSE / (len(corpusObj.m_transferLabelList) * len(self.ex_id_cs)))
                RMSE_list.append(RMSE)

            acc = self.get_pred_acc(fn_test, label_test, km_idx, p_idx, p_label, weakAnswers_dict, corpusObj)
            self.m_accList.append(acc)

            if rr % 20 == 0:
                print(rr, acc)
                print("RMSE: ", str(rr), str(RMSE_list))
                print("agg acc: ",str(rr), str(agg_acc_list), str(correct), str(total))

        print("finished!")

    def activeTrainClf(self, corpusObj):
        cost = 0.0

        if self.m_algType == "SPV":
            feature_train = corpusObj.m_feature[self.m_train]
            label_train = corpusObj.m_label[self.m_train]
            self.m_activeClf.fit(feature_train, label_train)
            cost += len(label_train)
        else:
            # initial: visit strong oracle
            initialInstanceNum = 3
            while True:
                initialIdx = random.sample(self.m_train, initialInstanceNum)
                label_init = corpusObj.m_label[initialIdx]
                if len(set(label_init)) > 1:
                    break

            # print("initExList\t", initialIdx, label_init)
            # no initial
            '''
            strongLabelNumIter = 0
            self.m_labeledIDList = []
            self.m_unlabeledIDList = list(set(self.m_train))
            
            # initial
            '''
            strongLabelNumIter = len(initialIdx)  # the number of visiting strong oracle

            self.m_labeledIDList.extend(initialIdx)
            self.m_unlabeledIDList = list(set(self.m_train) - set(initialIdx))

            strongAnswers_dict = {self.m_labeledIDList[i]: corpusObj.m_label[self.m_labeledIDList[i]] for i in
                                  range(len(self.m_labeledIDList))}

            feature_train_iter = corpusObj.m_feature[self.m_labeledIDList]
            label_train_iter = corpusObj.m_label[self.m_labeledIDList]

            self.m_activeClf.fit(feature_train_iter, label_train_iter)  # initial

            predLabelTest = self.m_activeClf.predict(corpusObj.m_feature[self.m_test])
            acc = accuracy_score(corpusObj.m_label[self.m_test], predLabelTest)
            print("strongLabelNumIter", strongLabelNumIter, "acc", acc)
            self.m_accList.append(acc)

            # Acc on gold
            acc_gold = self.m_activeClf.score(corpusObj.m_feature[self.m_gold], corpusObj.m_label[self.m_gold])
            self.m_accList_gold.append(acc_gold)

            print("strong label num threshold", self.m_strongLabelNumThresh)

            wk_total = 0
            wk_correct = 0

            visitedStaticOracle = False  # visit weak oracles only once

            cleanFeatureTrain = []
            cleanLabelTrain = []
            cleanIDTrain = []

            # id - order: selected by the classifier by margin-based strategy
            id_selectOrder = []
            id_conf = {}
            weakAnswers_dict = {}

            # gold - confidence - acc
            cur_cid = 0.5
            cf_gap = 0.1
            num_cf = int((1 - 0.5) / cf_gap) + 1
            confid_ids_weak_gold = {}
            acc_confid_weak_gold = {}
            id_conf = {}
            abandon_sz_iter = []

            # EM through iteration

            while strongLabelNumIter < self.m_strongLabelNumThresh and len(self.m_unlabeledIDList) > 0:
                idx = self.select_example(corpusObj)
                '''
                if strongLabelNumIter == 3:
                    idx = self.select_example(corpusObj)
                else:
                    idx = self.select_example_ByConf(id_conf)
                '''
                id_selectOrder.append(idx)

                # print(strongLabelNumIter, "idx", idx)
                self.m_labeledIDList.append(idx)
                self.m_unlabeledIDList.remove(idx)
                strongAnswers_dict[idx] = corpusObj.m_label[idx]
                #
                numRevisedAnsByWeak = 0  # the number of times that weak answers revise its previous answers
                numRevisedAnsByWeak_correct = 0  # the number of times that weak answers revise its previous answers CORRECTLY
                numRevisedAnsByStrong = 0  # the number of times that strong oracle revise the answer from weak oracles
                numRevisedAnsByClf = 0
                numRevisedAnsByClf_correct = 0

                if self.m_algType == "PLCS" and not visitedStaticOracle: #strongLabelNumIter == self.m_strongLabelNumThresh-1:
                    if self.m_useEM:
                        correct, total, cleanIDTrain, cleanLabelTrain, id_conf = self.generateCleanDataByCrowd_EM(corpusObj, self.m_train, dict(strongAnswers_dict))
                        #visitedStaticOracle = True  # only beginning
                    else:
                        correct, total, cleanIDTrain, cleanLabelTrain, id_conf = self.generateCleanDataByCrowd2(self.m_unlabeledIDList)
                        visitedStaticOracle = True  # only beginning

                        '''
                        # confidence (gap): ids
                        # from 0.5 to 1.0, gap=0.1
                        confid_ids = {round(0.5 + cf_gap * cf_i, 1): [] for cf_i in range(num_cf)}  # [0.5,0.6)
                        for wid in id_conf:
                            conf = id_conf[wid]
                            cid = round(int((conf-0.5)/cf_gap) * cf_gap + 0.5, 1)
                            confid_ids[cid].append(wid)

                        # gold confidence
                        correct_gold, total_gold, cleanIDTrain_gold, cleanLabelTrain_gold, id_conf_gold = self.generateCleanDataByCrowd2(
                            self.m_gold)
                        weakAnswers_gold_dict = {cleanIDTrain_gold[i]: cleanLabelTrain_gold[i] for i in range(len(cleanIDTrain_gold))}
                        confid_ids_weak_gold = {round(0.5 + cf_gap * cf_i, 1): [] for cf_i in range(num_cf)}  # [0.5,0.6): IDs
                        acc_confid_weak_gold = {round(0.5 + cf_gap * cf_i, 1): 0.0 for cf_i in range(num_cf)}  # [0.5, 0.6): acc on gold
                        for wid in id_conf_gold:
                            conf = id_conf_gold[wid]
                            cid = round(int((conf-0.5)/cf_gap) * cf_gap + 0.5, 1)
                            confid_ids_weak_gold[cid].append(wid)
                        for i in range(num_cf):
                            cid = round(i * cf_gap + 0.5, 1)
                            wkans_list = []
                            for id in confid_ids_weak_gold[cid]:
                                wkans_list.append(weakAnswers_gold_dict[id])
                            if len(confid_ids_weak_gold[cid]) > 0:
                                acc_confid_weak_gold[cid] = accuracy_score(corpusObj.m_label[confid_ids_weak_gold[cid]], wkans_list)
                        '''
                    cleanFeatureTrain = corpusObj.m_feature[cleanIDTrain]

                    # fully trust
                    '''
                    wk_total += total
                    wk_correct += correct
                    '''

                    # sub-label
                    wk_total = total
                    wk_correct = correct

                    wk_acc = 0.0
                    if wk_correct > 0:
                        wk_acc = float(wk_correct)/wk_total
                    if strongLabelNumIter % 50 ==0:
                        print("strong "+str(strongLabelNumIter)+" wk: "+str(wk_correct)+" "+str(wk_total)+" "+str(wk_acc))
                    self.m_correctAnsNum.append((strongLabelNumIter, wk_correct, wk_total, wk_acc))

                # sub-label     
                train_iter_dict = dict(strongAnswers_dict)

                if len(cleanFeatureTrain) > 0:
                    #  revise by strong oracle and classifier
                    weakAnswers_dict = {cleanIDTrain[i]: cleanLabelTrain[i] for i in range(len(cleanIDTrain))}

                    # classifier
                    acc_clf_cur = self.m_accList_gold[-1]
                    if acc_clf_cur >= 1.0:
                        label_clf_list = self.m_activeClf.predict(corpusObj.m_feature[cleanIDTrain])

                    for i in range(len(cleanIDTrain)):
                        id_x = cleanIDTrain[i]
                        # classifier
                        if id_x not in strongAnswers_dict:
                            train_iter_dict[id_x] = weakAnswers_dict[id_x]
                            # classifier
                            if acc_clf_cur >= 1.0:
                                label_clf_idx = label_clf_list[i]  # self.m_activeClf.predict(corpusObj.m_feature[id_x].reshape(1, -1))[0]
                                if label_clf_idx != weakAnswers_dict[id_x]:  # clf != weak
                                    train_iter_dict[id_x] = label_clf_idx  # revise by clf
                                    numRevisedAnsByClf += 1

                                    # revise correctly or not?
                                    label_true_idx = corpusObj.m_label[id_x]
                                    if label_clf_idx == label_true_idx:  # clf is correct
                                        numRevisedAnsByClf_correct += 1

                                    print("strongIter: " + str(
                                        strongLabelNumIter) + " classifier revised correctly: " + str(
                                        numRevisedAnsByClf_correct) + " clf revised: " +
                                          str(numRevisedAnsByClf) + " strong revised: " + str(numRevisedAnsByStrong))

                        # strong oracle
                        else:
                            if strongAnswers_dict[id_x] != weakAnswers_dict[id_x]:
                                numRevisedAnsByStrong += 1
                                
                self.m_revisedAnsNum.append((strongLabelNumIter, numRevisedAnsByClf_correct, numRevisedAnsByClf, numRevisedAnsByStrong))

                '''
                # change confidence
                while cur_cid <= 1.0:  # abandon training data [cur_cid, cur_cid + gap)
                    cur_goldIDs = confid_ids_weak_gold[cur_cid]

                    if len(cur_goldIDs) > 0:
                        acc_clf_gold = self.m_activeClf.score(corpusObj.m_feature[cur_goldIDs],
                                                          corpusObj.m_label[cur_goldIDs])
                    else:
                        acc_clf_gold = 1.0
                    acc_weak_gold = acc_confid_weak_gold[cur_cid]
                    if acc_clf_gold <= acc_weak_gold:
                        break
                    train_iter_dict = {id: train_iter_dict[id] for id in train_iter_dict if
                                       id in strongAnswers_dict or id_conf[id] >= cur_cid + cf_gap}
                    abandon_sz_iter.append((strongLabelNumIter, cur_cid, len(train_iter_dict)))
                    cur_cid += cf_gap
                    cur_cid = round(cur_cid, 1)
                '''

                # sub-label
                feature_train_iter = []
                label_train_iter = []
                for key_id in train_iter_dict:
                    feature_train_iter.append(corpusObj.m_feature[key_id])
                    label_train_iter.append(train_iter_dict[key_id])

                '''
                # fully trust weak labels
                # data for training the classifier
                if len(cleanFeatureTrain) > 0:
                    feature_train_iter = np.vstack((cleanFeatureTrain, corpusObj.m_feature[self.m_labeledIDList]))
                    label_train_iter = np.hstack((cleanLabelTrain, corpusObj.m_label[self.m_labeledIDList]))
                else:
                    feature_train_iter = corpusObj.m_feature[self.m_labeledIDList]
                    label_train_iter = corpusObj.m_label[self.m_labeledIDList]
                
                # update label/unlabeled list
                if strongLabelNumIter == initialInstanceNum:
                    #self.m_labeledIDList.extend(cleanIDTrain)
                    for i in cleanIDTrain:
                        self.m_unlabeledIDList.remove(i)
                '''

                self.m_activeClf.fit(feature_train_iter, label_train_iter)

                predLabelTest = self.m_activeClf.predict(corpusObj.m_feature[self.m_test])
                acc = accuracy_score(corpusObj.m_label[self.m_test], predLabelTest)

                strongLabelNumIter += 1
                #print("strongLabelNumIter", strongLabelNumIter, "acc", acc)
                self.m_accList.append(acc)

                # Acc on gold
                acc_gold = self.m_activeClf.score(corpusObj.m_feature[self.m_gold],
                                                  corpusObj.m_label[self.m_gold])
                self.m_accList_gold.append(acc_gold)
                # For Gold-task, update estimation for weak oracles with gold + label
                #self.estimateOracleParam(self.m_gold + self.m_labeledIDList)

            #print("abandon"+str(abandon_sz_iter))
            # compare confidence and selected order
            '''
            common = []
            id_sortedByConf = sorted(id_conf.items(), key=operator.itemgetter(1))
            #if len(id_selectOrder) == len(id_sortedByConf):
            id_selectOrder = id_selectOrder[1:]
            id_selectOrder_dict = {id_selectOrder[i]: i for i in range(len(id_selectOrder))}

            # id_select_conf = []
            for id in id_selectOrder_dict:
                correct_cur = False
                if weakAnswers_dict[id] == corpusObj.m_label[id]:
                    correct_cur = True
                self.m_id_select_conf_list.append((id, id_selectOrder_dict[id], id_conf[id], correct_cur))

            i=0.1
            while i<=1:
                id_conf_list = id_sortedByConf[:int(i*len(id_selectOrder))]
                id_conf_list = [j for (j, g) in id_conf_list]
                id_select_list = id_selectOrder[:int(i*len(id_selectOrder))]
                commonNum = len(set(id_conf_list).intersection(set(id_select_list)))
                common.append(commonNum)
                i += 0.1
            print("common: "+str(common))
            '''



def loadData(corpusObj, dataName):
    if dataName == "electronics":
        featureLabelFile = "../../dataset/processed_acl/processedBooksKitchenElectronics/TF/" + dataName

        featureMatrix, labelList = readFeatureLabel(featureLabelFile)

        featureMatrix = np.array(featureMatrix)
        labelArray = np.array(labelList)

        transferLabelFile = "../../dataset/processed_acl/processedBooksKitchenElectronics/TF/transferLabel_books--electronics.txt"
        auditorLabelList, transferLabelList, targetLabelList = readTransferLabel(transferLabelFile)
        transferLabelArray = np.array(transferLabelList)
        auditorLabelArray = np.array(auditorLabelList)

        #
        transferLabelArrayList = []
        transferLabelArrayList.append(transferLabelArray)

        multipleClassFlag = False
        initialExList = [[397, 1942, 200], [100, 1978, 657], [902, 788, 1370], [1688, 1676, 873], [1562, 1299, 617],
                         [986, 1376, 562], [818, 501, 1922], [600, 1828, 1622], [1653, 920, 1606], [39, 1501, 166]]

        #
        #a1 = labelList.count(1.0)
        #a2 = labelList.count(0.0)
        distinct_labels = list(set(labelArray))
        corpusObj.initCorpus(featureMatrix, labelArray, transferLabelArrayList, auditorLabelArray, initialExList, "text",
                             multipleClassFlag, distinct_labels)

    if dataName == "sensorTypes":
        raw_pt = [i.strip().split('\\')[-1][:-5] for i in
                  open('../../dataset/sensorType/rice_pt_soda').readlines()]
        #raw_pt = [i.strip().split('\\')[-1] for i in
        #         open('../../dataset/sensorType/sdh_soda_rice/rice_names').readlines()]
        fn = get_name_features(raw_pt)

        featureMatrix = fn

        featureMatrix = np.array(featureMatrix)
        # labelArray = np.array(labelList)

        transferLabelFile = "../../dataset/sensorType/sdh_soda_rice_new/transferLabel_sdh--rice--RFC.txt"
        auditorLabelList, transferLabelList, trueLabelList = readTransferLabel(transferLabelFile)

        auditorLabelArray = np.array(auditorLabelList)
        transferLabelArray = np.array(transferLabelList)
        labelArray = np.array(trueLabelList)


        transferLabelFile2 = "../../dataset/sensorType/sdh_soda_rice_new/transferLabel_sdh--rice--SVM.txt"
        auditorLabelList2, transferLabelList2, trueLabelList2 = readTransferLabel(transferLabelFile2)
        transferLabelArray2 = np.array(transferLabelList2)

        
        transferLabelFile3 = "../../dataset/sensorType/sdh_soda_rice_new/transferLabel_sdh--rice--LR.txt"
        auditorLabelList3, transferLabelList3, trueLabelList3 = readTransferLabel(transferLabelFile3)
        transferLabelArray3 = np.array(transferLabelList3)


        transferLabelFile4 = "../../dataset/sensorType/sdh_soda_rice_new/transferLabel_soda--rice--RFC.txt"
        auditorLabelList4, transferLabelList4, trueLabelList4 = readTransferLabel(transferLabelFile4)
        transferLabelArray4 = np.array(transferLabelList4)

        transferLabelFile5 = "../../dataset/sensorType/sdh_soda_rice_new/transferLabel_soda--rice--SVM.txt"
        auditorLabelList5, transferLabelList5, trueLabelList5 = readTransferLabel(transferLabelFile5)
        transferLabelArray5 = np.array(transferLabelList5)
        
        transferLabelFile6 = "../../dataset/sensorType/sdh_soda_rice_new/transferLabel_soda--rice--LR.txt"
        auditorLabelList6, transferLabelList6, trueLabelList6 = readTransferLabel(transferLabelFile6)
        transferLabelArray6 = np.array(transferLabelList6)


        #
        transferLabelArrayList = []

        transferLabelArrayList.append(transferLabelArray)
        transferLabelArrayList.append(transferLabelArray2)
        transferLabelArrayList.append(transferLabelArray3)
        transferLabelArrayList.append(transferLabelArray4)
        transferLabelArrayList.append(transferLabelArray5)
        transferLabelArrayList.append(transferLabelArray6)


        multipleClassFlag = True
        initialExList = [[470, 352, 217], [203, 280, 54], [267, 16, 190], [130, 8, 318], [290, 96, 418], [252, 447, 55],
                         [429, 243, 416], [240, 13, 68], [115, 449, 226], [262, 127, 381]]

        distinct_labels = list(set(trueLabelList))

        corpusObj.initCorpus(featureMatrix, labelArray, transferLabelArrayList, auditorLabelArray, initialExList, "sensor",
                             multipleClassFlag, distinct_labels, raw_pt)


def CVALParaWrapper(args):
    return CVALPerFold(*args)


def CVALPerFold(corpusObj, initialSampleList, gold_datast, weak_trn_dataset, train, test):
    StrongLabelNumThreshold = 120

    random.seed(10)
    np.random.seed(10)

    # for i in range(StrongLabelNumThreshold):

    # print(i, "random a number", random.random())
    # print(i, "numpy random a number", np.random.random())
    alObj = _ActiveClf(corpusObj.m_category, corpusObj.m_multipleClass, corpusObj.m_labelNum, StrongLabelNumThreshold)
    alObj.initActiveClf(initialSampleList, gold_datast, weak_trn_dataset, train, test)
    #alObj.activeTrainClf(corpusObj)
    #alObj.activeTrainClf_al(corpusObj)
    #alObj.activeTrainClf_global(corpusObj)
    #alObj.activeTrainClf_cluster(corpusObj)
    alObj.activeTrainClf_cluster_old(corpusObj)
    #alObj.expeiment_1(corpusObj)

    accList = alObj.m_accList

    resultPerFold = []
    resultPerFold.append(accList)
    resultPerFold.append(alObj.m_correctAnsNum)
    resultPerFold.append(alObj.m_weakOracleAcc_pred)
    resultPerFold.append(alObj.m_weakOracleAcc_true)
    resultPerFold.append(alObj.m_rmsePredWeakAcc)
    resultPerFold.append(alObj.m_revisedAnsNum)
    resultPerFold.append(alObj.m_id_select_conf_list)
    resultPerFold.append(alObj.m_prior_pred)
    resultPerFold.append(alObj.m_prior_true)
    resultPerFold.append(alObj.m_accList_al)

    return resultPerFold


def parallelCVAL(corpusObj, outputSrc, modelVersion):
    totalSampleNum = len(corpusObj.m_label)
    print("number of samples in dataset:", totalSampleNum)
    sampleIndexList = [i for i in range(totalSampleNum)]
    random.shuffle(sampleIndexList)

    # gold rate
    goldRate = 0.0#0.15
    goldRate2 = 1.0
    # weak oracle training dataset
    weakRate = 0.0# 0.4

    foldNum = 10
    perFoldSampleNum = int(totalSampleNum * 1.0 / foldNum)
    foldSampleList = []

    for foldIndex in range(foldNum - 1):
        perFoldSampleList = sampleIndexList[foldIndex * perFoldSampleNum:(foldIndex + 1) * perFoldSampleNum]
        foldSampleList.append(perFoldSampleList)

    perFoldSampleList = sampleIndexList[perFoldSampleNum * (foldNum - 1):]
    foldSampleList.append(perFoldSampleList)

    totalAccList = [[] for i in range(foldNum)]

    #
    totalOracleAccPredList = [[] for i in range(foldNum)]  # pred by gold task / EM
    totalOracleAccTrueList = [[] for i in range(foldNum)]  # True acc on untrained data
    totalCorrectWeakAnsNumList = [[] for i in range(foldNum)]  #
    totalRMSEOracleAccPred = [[] for i in range(foldNum)]
    totalRevisedAnsList = [[] for i in range(foldNum)]
    totalIdSelectConf = [[] for i in range(foldNum)]
    totalClassPriorPred = [[] for i in range(foldNum)]
    totalClassPriorTrue = [[] for i in range(foldNum)]

    totalAccList_al = [[] for i in range(foldNum)]


    totalWeakCoverList = [[] for i in range(foldNum)]  # confidence > threshold
    totalWeakLabelAccList = [[] for i in range(foldNum)]
    totalWeakLabelPrecisionList = [[] for i in range(foldNum)]
    totalWeakLabelRecallList = [[] for i in range(foldNum)]

    # totalWeakLabelNumList = [[] for i in range(foldNum)]

    # totalSampleNum4PosAuditorList = [[] for i in range(foldNum)]
    # totalSampleNum4NegAuditorList = [[] for i in range(foldNum)]

    poolNum = 10

    results = []
    argsList = [[] for i in range(poolNum)]

    # supervised acc
    acc_supervised = []
    acc_supervised_trn = []

    # training dataset, gold dataset, weak oracle datase, test dataset
    for poolIndex in range(poolNum):
        foldIndex = poolIndex
        train = []
        for preFoldIndex in range(foldIndex):
            train.extend(foldSampleList[preFoldIndex])

        test = foldSampleList[foldIndex]
        for postFoldIndex in range(foldIndex + 1, foldNum):
            train.extend(foldSampleList[postFoldIndex])

        argsList[poolIndex].append(corpusObj)

        initialSampleList = corpusObj.m_initialExList[foldIndex]
        argsList[poolIndex].append(initialSampleList)

        # extract gold dataset from train dataset
        gold_dataset = random.sample(train, int(goldRate * len(train)))
        train = [i for i in train if i not in gold_dataset]

        # weak oracle training dataset
        weak_trn_dataset = random.sample(train, int(weakRate * len(train)))
        train = [i for i in train if i not in weak_trn_dataset]

        gold_dataset2 = random.sample(gold_dataset, int(goldRate2 * len(gold_dataset)))

        argsList[poolIndex].append(gold_dataset2)
        argsList[poolIndex].append(weak_trn_dataset)

        argsList[poolIndex].append(train)
        argsList[poolIndex].append(test)

        # supervised learning

        if corpusObj.m_labelNum > 2:
            superClf = LR(multi_class="multinomial", solver='lbfgs', random_state=3, fit_intercept=False)
        else:
            superClf = LR(random_state=3)
        trn_feature = corpusObj.m_feature[train]
        trn_label = corpusObj.m_label[train]
        superClf.fit(trn_feature, trn_label)
        tst_feature = corpusObj.m_feature[test]
        tst_label = corpusObj.m_label[test]
        acc_sp = superClf.score(tst_feature, tst_label)
        acc_supervised.append(acc_sp)
        acc_sp_trn = superClf.score(trn_feature, trn_label)
        acc_supervised_trn.append(acc_sp_trn)

    ave_acc_supervised = sum(acc_supervised)/len(acc_supervised)
    print("supervised average acc on test: "+str(ave_acc_supervised))
    ave_acc_spv_trn = sum(acc_supervised_trn)/len(acc_supervised_trn)
    print("supervised average acc on train: " + str(ave_acc_spv_trn))
    # the main process with multiprocessing
    poolObj = Pool(poolNum)
    results = poolObj.map(CVALParaWrapper, argsList)
    poolObj.close()
    poolObj.join()
    # results = map(CVALParaWrapper, argsList)

    for poolIndex in range(poolNum):
        foldIndex = poolIndex
        resultFold = results[foldIndex]
        totalAccList[foldIndex] = resultFold[0]
        totalCorrectWeakAnsNumList[foldIndex] = resultFold[1]
        totalOracleAccPredList[foldIndex] = resultFold[2]
        totalOracleAccTrueList[foldIndex] = resultFold[3]
        totalRMSEOracleAccPred[foldIndex] = resultFold[4]
        totalRevisedAnsList[foldIndex] = resultFold[5]
        totalIdSelectConf[foldIndex] = resultFold[6]
        totalClassPriorPred[foldIndex] = resultFold[7]
        totalClassPriorTrue[foldIndex] = resultFold[8]
        totalAccList_al[foldIndex] = resultFold[9]


    # print(len(accList))
    # print(accList)

    # for foldIndex in range(foldNum):
    # 	train = []
    # 	for preFoldIndex in range(foldIndex):
    # 		train.extend(foldSampleList[preFoldIndex])

    # 	test = foldSampleList[foldIndex]
    # 	for postFoldIndex in range(foldIndex+1, foldNum):
    # 		train.extend(foldSampleList[postFoldIndex])

    # 	initialSampleList = corpusObj.m_initialExList[foldIndex]

    # 	alObj = _ActiveClf(corpusObj.m_category, corpusObj.m_multipleClass, StrongLabelNumThreshold)
    # 	alObj.initActiveClf(initialSampleList, train, test)
    # 	alObj.activeTrainClf(corpusObj)

    # totalAccList[foldIndex] = alObj.m_accList

    writeFile(outputSrc, modelVersion, totalAccList, "acc")
    writeFile(outputSrc, modelVersion, totalCorrectWeakAnsNumList, "correct")
    writeFile(outputSrc, modelVersion, totalOracleAccPredList, "oracleAccPred")
    writeFile(outputSrc, modelVersion, totalOracleAccTrueList, "oracleAccTrue")
    writeFile(outputSrc, modelVersion, totalRMSEOracleAccPred, "oracleAccPredRMSE")
    writeFile(outputSrc, modelVersion, totalRevisedAnsList, "revisedAns")
    writeFile(outputSrc, modelVersion, totalIdSelectConf, "idSelectConf")
    #writeFile(outputSrc, modelVersion, totalClassPriorPred, "classPriorPred")
    #writeFile(outputSrc, modelVersion, totalClassPriorTrue, "classPriorTrue")
    writeFile(outputSrc, modelVersion, totalAccList_al, "acc_al")

    # writeFile(outputSrc, modelVersion, totalAccList, "acc")
    # writeFile(outputSrc, modelVersion, totalWeakLabelAccList, "weakLabelAcc")
    # writeFile(outputSrc, modelVersion, totalWeakLabelPrecisionList, "weakLabelPrecision")
    # writeFile(outputSrc, modelVersion, totalWeakLabelRecallList, "weakLabelRecall")
    # writeFile(outputSrc, modelVersion, totalWeakLabelNumList, "weakLabelNum")


if __name__ == '__main__':
    timeStart = datetime.now()

    corpusObj = _Corpus()
    #dataName = "electronics"
    dataName = "sensorTypes"
    loadData(corpusObj, dataName)

    modelName = "random_" + dataName
    timeStamp = datetime.now()
    timeStamp = str(timeStamp.month) + str(timeStamp.day) + str(timeStamp.hour) + str(timeStamp.minute)
    modelVersion = modelName + "_" + timeStamp
    fileSrc = dataName

    # CVAL(corpusObj, fileSrc, modelVersion)
    parallelCVAL(corpusObj, fileSrc, modelVersion)
    timeEnd = datetime.now()
    print("duration", (timeEnd - timeStart).total_seconds())