'''
 * IPSN2020 Selective Sampling for Sensor Type Classification in Buildings
 * Author: Jing Ma
 * Date:2020-02
'''

import argparse
import numpy as np
import math
import random
import re
import itertools
import operator
import sys

from collections import defaultdict as dd
from collections import Counter as ct

from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer as TV
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

parser = argparse.ArgumentParser()

# rice
parser.add_argument('--target', type=str, default='rice', help='target buildings')
parser.add_argument('--budget', type=int, default=100, help='budget')
parser.add_argument('--initK', type=int, default=11, help='initial cluster num')
parser.add_argument('--initK_AL', type=int, default=11, help='initial cluster num for AL')
parser.add_argument('--updateAns', type=int, default=1, help='update weak answers')
parser.add_argument('--updateCluster', type=int, default=1, help='update cluster')
parser.add_argument('--labelPropag', type=int, default=1, help='label propagation')
parser.add_argument('--select', type=str, default='disagree6', help='selection strategy')
parser.add_argument('--confid_thresh', type=float, default=0.9, help='confidence threshold')
parser.add_argument('--iter_update', type=int, default=1, help='update answer per x iteration')
parser.add_argument('--iter_cluster', type=int, default=20, help='update cluster per x iteration')
parser.add_argument('--r', type=float, default=2, help='entropy threshold for updating cluster')
parser.add_argument('--postfix', type=str, default='', help='output')
parser.add_argument('--wv', type=int, default=1, help='wv')

args = parser.parse_args()


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
        # self.m_transferLabel = None
        self.m_transferLabelList = []
        self.m_auditorLabel = None
        self.m_initialExList = []
        #
        self.m_sensorName = {}
        self.m_labelNum = None
        self.m_label2Idx = {}
        self.m_idx2Label = {}
        self.m_accTransferList = []
        self.m_transferAll = {}  # TL

    def initCorpus(self, featureMatrix, labelArray, transferLabelArrayList, auditorLabelArray, initialExList, category,
                   multipleClass, distinct_label_list, sensorName=None):
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
        print("transfer acc: " + str(self.m_accTransferList))

        # calculate the variance of accuracy in class-level
        # class prior
        instancenum_inClass = [0 for i in range(self.m_labelNum)]
        class_instances = {i: [] for i in range(self.m_labelNum)}
        for id in range(len(corpusObj.m_label)):
            lb = corpusObj.m_label[id]
            lb_id = corpusObj.m_label2Idx[lb]
            instancenum_inClass[lb_id] += 1
            class_instances[lb_id].append(id)

        var_of_acc_all = []
        for i in range(len(transferLabelArrayList)):
            var_of_acc = 0.0
            mean_acc = self.m_accTransferList[i]
            for lb_id in range(self.m_labelNum):
                lb = self.m_idx2Label[lb_id]
                ids_class = class_instances[lb_id]
                class_acc = accuracy_score(labelArray[ids_class], transferLabelArrayList[i][ids_class])
                class_size = instancenum_inClass[lb_id]
                var_of_acc += (math.pow(class_acc - mean_acc, 2) * class_size)

            var_of_acc /= len(corpusObj.m_label)
            # var_of_acc /= corpusObj.m_labelNum
            var_of_acc = math.sqrt(var_of_acc)
            var_of_acc_all.append(var_of_acc)
        print("variance of accuracy for transferred labels: ", var_of_acc_all)
        print()


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

        self.m_transferAcc_pred_hist = []
        self.m_transferAcc_true_hist = []

        self.m_initParamWithGold = True
        self.m_algType = "PLCS"  # supervised, AL, PL, PLCS

        self.m_correctAnsNum = []  # (correctNum, ansNum) by weak
        self.m_revisedAnsNum = []  # (clfCorrectRevise, clfRevised, StrongRevise)

        self.m_id_select_conf_list = []
        # self.m_weakLabelPrecisionList = []
        # self.m_weakLabelRecallList = []
        # self.m_weakLabelAccList = []  # acc when answer

        self.m_weakOracleCM_cluster_pred = {}  # {cluster:{oracle: CM}}
        self.m_weakOracleAcc_cluster_true = {}
        self.m_weakOracleAcc_cluster_pred = {}  # {cluster:{oracle: acc}}

        self.m_accTransfer_cluster_pred = {}
        self.m_accTransfer_cluster_true = {}

        self.m_accList_al = []

        #
        self.tao = 0
        self.alpha_ = 1

        self.clf = LinearSVC()
        self.clf_al = LinearSVC()
        self.ex_id = dd(list)
        self.ex_id_cs = dd(list)

        self.id2cluster = {}
        self.m_crowdAcc = []  # accuracy of crowdsourcing

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
            weak_rate = []
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
                    # self.m_accTransfer_pred = acc_true_list[:partOracleStart]  # will not be used, will be changed
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
                            self.m_accTransfer_pred.append(
                                accuracy_score(gold_labels, corpusObj.m_transferLabelList[i][self.m_gold]))
                            RMSE += math.pow(corpusObj.m_accTransferList[i] - self.m_accTransfer_pred[i], 2)

                    RMSE = math.sqrt(RMSE / nObservers)
                    print("RMSE: " + str(RMSE))
                    self.m_rmsePredWeakAcc.append(RMSE)

        # classifier
        if self.m_multipleClass:
            self.clf = LR(multi_class="multinomial", solver='lbfgs', random_state=3, fit_intercept=False)
            self.clf_al = LR(multi_class="multinomial", solver='lbfgs', random_state=3, fit_intercept=False)
        else:
            # self.m_activeClf = LR(random_state=3)
            self.clf = LR(random_state=3)
            self.clf_al = LR(random_state=3)


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

    def computeSumSqure(self, acc_pred, acc_true):
        ss = 0.0
        for i in range(len(acc_pred)):
            ss += math.pow(acc_pred[i] - acc_true[i], 2)
        return ss


    def update_tao(self, labeled_set, corpusObj):

        dist_inter = []
        pair = list(itertools.combinations(labeled_set, 2))

        for p in pair:

            d = np.linalg.norm(corpusObj.m_feature[p[0]] - corpusObj.m_feature[p[1]])
            if corpusObj.m_label[p[0]] != corpusObj.m_label[p[1]]:
                dist_inter.append(d)

        try:
            self.tao = self.alpha_ * min(dist_inter) / 2  # set tao be the min(inter-class pair dist)/2
        except Exception as e:
            self.tao = self.tao

    def findNN(self, src, example_list):  # find nearest ex to src in unlabled set
        # use a matrix is faster than using a for loop

        unlabeled_ex = list(
            set(example_list) - set([src]))  # TODO: updated exmaple_list with the correct var name for list of examples
        fea = corpusObj.m_feature[unlabeled_ex] - corpusObj.m_feature[src]  # assuming this indexing returns nparrays
        dist = np.linalg.norm(fea, axis=1)

        ix_min = np.argmin(dist)
        nearest_id = unlabeled_ex[ix_min]
        return nearest_id

    def update_pseudo_set(self, new_ex_id, cluster_id, p_idx, p_label, p_dist, p_src, corpusObj):

        tmp = []
        idx_tmp = []
        label_tmp = []

        current_ids_in_cluster = [id for cid in self.ex_id for id in self.ex_id[cid]]

        # re-visit pseudo-labeled exs on previous itr with the new tao
        for i, j in zip(p_idx, p_label):

            if p_dist[i] < self.tao:
                idx_tmp.append(i)
                label_tmp.append(j)
            else:
                p_dist.pop(i)
                tmp.append(i)

        # put the removed exs back to where it came from
        for id in tmp:
            if p_src[id] in self.ex_id:
                c_id = p_src[id]
            else:
                nn = self.findNN(id, current_ids_in_cluster)  # the nearest unlabeled neighbor
                for c in self.ex_id.keys():
                    if nn in self.ex_id[c]:
                        c_id = c
                        break

            self.ex_id[c_id].append(id)

            p_src.pop(id)

        p_idx = idx_tmp
        p_label = label_tmp
        tmp = []

        # added exs to pseudo set
        if cluster_id in self.ex_id:
            for ex in self.ex_id[cluster_id]:

                if ex == new_ex_id:
                    continue
                d = np.linalg.norm(corpusObj.m_feature[ex] - corpusObj.m_feature[new_ex_id])

                if d < self.tao:
                    p_dist[ex] = d
                    p_src[ex] = cluster_id
                    p_idx.append(ex)
                    p_label.append(corpusObj.m_label[new_ex_id])
                else:
                    tmp.append(ex)

            if not tmp:
                self.ex_id.pop(cluster_id)
            else:
                self.ex_id[cluster_id] = tmp

        return p_idx, p_label, p_dist, p_src

    def computeAveClusterEntropy(self, ex_id, corpusObj, clf=None):
        if clf is None:
            clf = self.clf

        sub_pred = dd(list)  # Mn predicted labels for each cluster

        for k, v in ex_id.items():
            sub_pred[k] = clf.predict(corpusObj.m_feature[v])  # predict labels for cluster learning set

        # entropy-based cluster selection
        H_all = 0.0
        for k, v in sub_pred.items():
            count = ct(v).values()
            count[:] = [i / float(max(count)) for i in count]
            H = np.sum(-p * math.log(p, 2) for p in count if p != 0)
            H_all += H

        H_ave = H_all / len(ex_id)
        return H_ave

    def select_example_cluster(self, labeled_set, corpusObj, clf=None):
        if clf is None:
            clf = self.clf

        sub_pred = dd(list)  # Mn predicted labels for each cluster
        idx = 0
        # initialize: find the first instance which is not in labeled dataset
        for id in self.m_train:
            if id not in labeled_set:
                idx = id
                break

        for k, v in self.ex_id.items():
            sub_pred[k] = clf.predict(corpusObj.m_feature[v])  # predict labels for cluster learning set

        # entropy-based cluster selection
        rank = []
        for k, v in sub_pred.items():
            count = ct(v).values()
            count[:] = [i / float(max(count)) for i in count]
            H = np.sum(-p * math.log(p, 2) for p in count if p != 0)
            rank.append([k, len(v), H])
        rank = sorted(rank, key=lambda x: x[-1], reverse=True)

        if not rank:
            raise ValueError('no clusters found in this iteration!')

        c_idx = rank[0][0]  # pick the 1st cluster on the rank, ordered by label entropy
        c_ex_id = self.ex_id[c_idx]  # examples in the cluster picked
        sub_label = sub_pred[c_idx]  # used when choosing cluster by H
        sub_fn = corpusObj.m_feature[c_ex_id]

        # sub-cluster the cluster
        c_ = KMeans(init='k-means++', n_clusters=len(np.unique(sub_label)), n_init=10)
        c_.fit(sub_fn)
        dist = np.sort(c_.transform(sub_fn))

        ex_ = dd(list)  # clusterid, id, dist, predicted label
        for i, j, k, l in zip(c_.labels_, c_ex_id, dist, sub_label):
            ex_[i].append([j, l, k[0]])
        for i, j in ex_.items():  # sort by ex. dist to the centroid for each C
            ex_[i] = sorted(j, key=lambda x: x[-1])
        for k, v in ex_.items():

            if v[0][0] not in labeled_set:  # find the first unlabeled ex

                idx = v[0][0]

                c_ex_id.remove(idx)  # update the training set by removing selected ex id

                if len(c_ex_id) == 0:
                    self.ex_id.pop(c_idx)
                else:
                    self.ex_id[c_idx] = c_ex_id
                break

        return idx, c_idx

    def sub_cluster(self, cid, clf):
        if cid not in self.ex_id:
            return

        c_ex_id = self.ex_id[cid]
        sub_label = clf.predict(corpusObj.m_feature[c_ex_id])
        sub_fn = corpusObj.m_feature[c_ex_id]

        c_ = KMeans(init='k-means++', n_clusters=len(np.unique(sub_label)), n_init=10)
        c_.fit(sub_fn)

        self.ex_id.pop(cid)

        if not self.ex_id:
            new_clusterid = 1
        else:
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
                    lb = corpusObj.m_label[id]  # labeled
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

    def computeKL(self, lb_pre_al, lb_pre):
        KL = 0.0
        for i in range(len(lb_pre_al)):
            p_al = lb_pre_al[i]
            p_cs = lb_pre[i]
            KL += p_al * math.log(p_al / p_cs, math.e)

        return KL

    def computeKLByPartPred(self, corpusObj, clf, clf_al, lb_pre, lb_pre_al, alpha=0.01):
        clf_class = clf.classes_
        clf_al_class = clf_al.classes_

        pred_distri_clf = np.zeros(corpusObj.m_labelNum)
        pred_distri_clf_al = np.zeros(corpusObj.m_labelNum)

        # smooth
        pred_distri_clf = pred_distri_clf + alpha
        pred_distri_clf_al = pred_distri_clf_al + alpha

        for lb_i in range(len(clf_class)):
            lb = clf_class[lb_i]
            lb_id = corpusObj.m_label2Idx[lb]  # label id in all classes (11)
            pred_distri_clf[lb_id] += lb_pre[lb_i]
        pred_distri_clf = pred_distri_clf / pred_distri_clf.sum()

        for lb_i in range(len(clf_al_class)):
            lb = clf_al_class[lb_i]
            lb_id = corpusObj.m_label2Idx[lb]  # label id in all classes (11)
            pred_distri_clf_al[lb_id] += lb_pre_al[lb_i]

        pred_distri_clf_al = pred_distri_clf_al / pred_distri_clf_al.sum()

        KL = self.computeKL(pred_distri_clf_al, pred_distri_clf) + self.computeKL(pred_distri_clf, pred_distri_clf_al)

        return KL

    # for id in {disagree}, sort by: kl * |neigbbor in disagree|
    def select_example_weak_KL_6(self, labeled_set, corpusObj, weakAns_dict, clf=None):
        if clf is None:
            clf = self.clf
        clf_al = self.clf_al

        # disagreement between clf_al and weak answers
        # choose the one with highest KL sum among neighbors

        neighborNum = {}
        id_disagree = {}  # id: disagree or not
        id_KL = {}

        # compute once: prediction for all the instances in train set
        id2rank = {self.m_train[i]: i for i in range(len(self.m_train))}

        lb_al_list = clf_al.predict(corpusObj.m_feature[self.m_train])
        lb_list = clf.predict(corpusObj.m_feature[self.m_train])
        lb_al_list_prob = clf_al.predict_proba(corpusObj.m_feature[self.m_train])
        lb_list_prob = clf.predict_proba(corpusObj.m_feature[self.m_train])

        lb_pre_al_lb_dict = {id: lb_al_list[id2rank[id]] for id in self.m_train}  # predicted answer by clf_AL
        lb_pre_lb_dict = {id: lb_list[id2rank[id]] for id in self.m_train}  # predicted answer by clf

        lb_pre_al_dict = {id: lb_al_list_prob[id2rank[id]] for id in self.m_train}
        lb_pre_dict = {id: lb_list_prob[id2rank[id]] for id in self.m_train}

        for id in self.m_train:

            if id in labeled_set:
                continue
            cluster_id = self.id2cluster[id]

            # disagree?
            lb_pre_al_lb = lb_pre_al_lb_dict[id]  # predicted answer  by AL
            lb_pre_lb = lb_pre_lb_dict[id]

            if lb_pre_al_lb != lb_pre_lb:  # disagree
                id_disagree[id] = 1
            else:
                continue

            # kl
            lb_pre_al = lb_pre_al_dict[id]  # predicted answer distribution by AL
            lb_pre = lb_pre_dict[id]

            KL = self.computeKLByPartPred(corpusObj, clf, clf_al, lb_pre, lb_pre_al)

            id_KL[id] = KL

            # find id's neighbors

            neighborNum[id] = len(self.ex_id_cs[cluster_id])

        # filter those has different predictions!
        id_KL = {id: id_KL[id] * neighborNum[id] for id in id_KL if id in id_disagree}

        # sort by KL sum
        id2c = {id: cid for cid in self.ex_id for id in self.ex_id[cid]}  # cluster for AL
        if len(id_disagree) > 0:
            sort_kl_sum = sorted(id_KL.items(), key=lambda x: x[1], reverse=True)
            idx = sort_kl_sum[0][0]
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

    def get_pred_acc(self, fn_test, label_test, labeled_set, pseudo_set, pseudo_label, weakAns_dict, corpusObj,
                     clf=None):
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
            labelID_set = set(labeled_set).union(set(pseudo_set))  # set of labeled and pseudo-labeled ID
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

    def updateTrueTransAcc_cluster(self, ex_id):
        self.m_accTransfer_cluster_true = {}  # [cid][oi]
        for cid in ex_id:
            ids_c = ex_id[cid]
            acc_c = []
            for oi in range(len(corpusObj.m_transferLabelList)):
                correct_oi = 0
                for idx in ids_c:
                    transferLabel = corpusObj.m_transferLabelList[oi][idx]
                    gt_label = corpusObj.m_label[idx]
                    if int(transferLabel) == int(gt_label):
                        correct_oi += 1
                acc_oi_c = float(correct_oi) / len(ids_c)
                acc_c.append(acc_oi_c)

            self.m_accTransfer_cluster_true[cid] = acc_c
        return

    def IWMV_cluster(self, ex_id, fix_answers, max_iter=500, tol=0.00001):
        predictY = {}  # id: label
        id_conf = {}
        weight_all = {}

        for cid in ex_id:
            instances = ex_id[cid]
            vote_weight = np.ones(len(corpusObj.m_transferLabelList))
            weight = np.zeros(len(corpusObj.m_transferLabelList))

            converged = False
            old_weight = weight.copy()

            for iter in range(max_iter):
                # e-step: y^
                for idx in instances:
                    if idx in fix_answers:
                        predictY[idx] = fix_answers[idx]
                        id_conf[idx] = 1.0
                    else:
                        votes = [0.0 for i in range(corpusObj.m_labelNum)]
                        for i in range(len(corpusObj.m_transferLabelList)):
                            transferLabel = corpusObj.m_transferLabelList[i][idx]
                            ans_idx = corpusObj.m_label2Idx[transferLabel]
                            votes[ans_idx] += vote_weight[i]

                        confidence = max(votes) / sum(votes)
                        predictY[idx] = corpusObj.m_idx2Label[votes.index(max(votes))]
                        id_conf[idx] = confidence

                # m-step: w, v
                for i in range(len(corpusObj.m_transferLabelList)):
                    weight[i] = float(sum([1 for idx in instances if
                                           corpusObj.m_transferLabelList[i][idx] == predictY[idx]])) / len(
                        instances)
                    vote_weight[i] = weight[i] * corpusObj.m_labelNum - 1

                # converge
                weight_diff = np.sum(np.abs(weight - old_weight))
                if weight_diff < tol:
                    converged = True
                    break

                old_weight = weight.copy()

            weight_all[cid] = weight.copy()

        return predictY, weight_all, id_conf

    def computeAccRMSE_cluster(self, trueAcc_cluster, predAcc_cluster):
        RMSE = 0.0
        for cid in trueAcc_cluster:
            RMSE += self.computeSumSqure(trueAcc_cluster[cid], predAcc_cluster[cid])

        c_num = len(trueAcc_cluster)
        oracle_num = len(corpusObj.m_transferLabelList)

        RMSE = math.sqrt(RMSE / (c_num * oracle_num))
        return RMSE

    def activeTrainClf_cluster_new(self, corpusObj):

        seed = 111
        label_propag = args.labelPropag  # enable label propagation or not
        flag_sub_cluster = args.updateCluster
        flag_update_agg = args.updateAns

        selection_strategy = args.select
        iter_up_ans = args.iter_update
        iter_cluster = args.iter_cluster

        r_thresh = args.r

        init_K = args.initK
        init_K_AL = args.initK_AL

        trainSampleNum = []
        # weak oracle answers
        self.m_train.sort()

        responses = self.gatherAnsFromWeakOracles(self.m_train, [])

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

        ex_id_cs1, id2cluster1 = self.clusterWithFeatures(init_K, self.m_train, feature1, seed)

        # statistics
        self.ex_id_cs = ex_id_cs1
        self.id2cluster = id2cluster1

        averageH_hist = []
        clusterChange = [0]  # iteration that cluster changes
        averageH_hist.append(0.0)

        # update true acc per cluster
        self.updateTrueTransAcc_cluster(self.ex_id_cs)
        self.m_transferAcc_true_hist.append((0, self.m_accTransfer_cluster_true))

        #print(self.m_accTransfer_cluster_true)

        # crowdsourcing by cluster
        confid_thresh = args.confid_thresh

        fix_answers = {}

        # EM+WV
        weakAnswers_dict, weight, id_conf = self.IWMV_cluster(self.ex_id_cs, fix_answers)

        weakAnswers_dict = {id: weakAnswers_dict[id] for id in weakAnswers_dict if
                            id_conf[id] >= confid_thresh}


        # oracle weight estimation
        self.m_transferAcc_pred_hist.append((0, weight))
        rmse_cur = self.computeAccRMSE_cluster(self.m_accTransfer_cluster_true, weight)
        self.m_rmsePredWeakAcc.append(rmse_cur)


        # copy crowdsourcing cluster
        ex_id_tmp, id2cluster_tmp = self.clusterWithFeatures(init_K_AL, self.m_train, feature1, seed)

        self.ex_id = {cid: list(ex_id_tmp[cid]) for cid in ex_id_tmp}  #self.ex_id_cs.copy()
        ex_N = [[cid, len(self.ex_id[cid])] for cid in self.ex_id]
        ex = dd(list)
        for cid in self.ex_id:
            c = KMeans(init='k-means++', n_clusters=1, n_init = 10, random_state=seed)
            c.fit(corpusObj.m_feature[self.ex_id[cid]])
            dist = np.sort(c.transform(corpusObj.m_feature[self.ex_id[cid]]))  # size: instanceNum x clusterNum
            for id, dis in zip(self.ex_id[cid], dist):
                ex[cid].append([id, dis[0]])
            ex[cid] = sorted(ex[cid], key=lambda x: x[-1])  # sort instances in one cluster by distance
        ex_N = sorted(ex_N, key=lambda x: x[-1], reverse=True) # sorted by the size of clusters

        km_idx = []
        p_idx = []
        p_label = []
        p_dist = dd()
        p_src = dd()

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

            # if len(set(init_lbs)) < 2:
            if ctr < 3:  # \ get at least instances for initialization
                continue

            if label_propag:
                self.update_tao(km_idx, corpusObj)

            p_idx, p_label, p_dist, p_src = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist, p_src, corpusObj)

            # weak

            if ctr % iter_up_ans == 0:
                # fix answers
                if flag_update_agg:
                    fix_answers = self.getfixAns(km_idx, p_idx, p_label)

                    weakAnswers_dict, weight, id_conf = self.IWMV_cluster(self.ex_id_cs, fix_answers)

                    weakAnswers_dict = {id: weakAnswers_dict[id] for id in weakAnswers_dict if
                                        id_conf[id] >= confid_thresh}

                    # oracle weight estimation
                    self.m_transferAcc_pred_hist.append((ctr, weight))
                    rmse_cur = self.computeAccRMSE_cluster(self.m_accTransfer_cluster_true, weight)
                    self.m_rmsePredWeakAcc.append(rmse_cur)


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
                print(exc_type, e.args, fname, 'on line ' + str(exc_tb.tb_lineno))
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
                print(exc_type, e.args, fname, 'on line ' + str(exc_tb.tb_lineno))
                acc_al = np.nan
            self.m_accList_al.append(acc_al)
            trainSampleNum.append(
                (ctr, trainNum - trainNum_al, train_correct - train_correct_al, trainNum_al, train_correct_al))

            # class entropy
            averageH = self.computeAveClusterEntropy(self.ex_id_cs, corpusObj, self.clf)
            averageH_hist.append(averageH)

        select_ins_info.append((0, init_lbs))

        # fix answers
        if flag_update_agg:
            fix_answers = self.getfixAns(km_idx, p_idx, p_label)

            weakAnswers_dict, weight, id_conf = self.IWMV_cluster(self.ex_id_cs, fix_answers)

            weakAnswers_dict = {id: weakAnswers_dict[id] for id in weakAnswers_dict if
                                id_conf[id] >= confid_thresh}

            # oracle weight estimation
            self.m_transferAcc_pred_hist.append((ctr, weight))
            rmse_cur = self.computeAccRMSE_cluster(self.m_accTransfer_cluster_true, weight)
            self.m_rmsePredWeakAcc.append(rmse_cur)

        # class entropy
        averageH = self.computeAveClusterEntropy(self.ex_id_cs, corpusObj, self.clf)
        averageH_hist.append(averageH)

        old_aveH = averageH

        cl_id = []  # track cluster id on each iter
        ex_al = []  # track ex added on each iter
        fn_test = corpusObj.m_feature[self.m_test]
        label_test = corpusObj.m_label[self.m_test]
        for rr in range(ctr, self.m_strongLabelNumThresh):
            if not p_idx:
                fn_train = corpusObj.m_feature[km_idx]
                label_train = corpusObj.m_label[km_idx]
            else:
                fn_train = corpusObj.m_feature[np.hstack((km_idx, p_idx))]
                label_train = np.hstack((corpusObj.m_label[km_idx], p_label))

            fn_train, label_train = self.addWeakAns(km_idx, p_idx, p_label, weakAnswers_dict, corpusObj)
            self.clf.fit(fn_train, label_train)
            # acc of clf trained on only on al
            fn_train_al, label_train_al = self.addWeakAns(km_idx, p_idx, p_label, dict(), corpusObj)
            self.clf_al.fit(fn_train_al, label_train_al)

            fix_answers = self.getfixAns(km_idx, p_idx, p_label)

            # if rr < 60:
            if selection_strategy == 'entropy':
                idx, c_idx, = self.select_example_cluster(km_idx, corpusObj, self.clf_al)
            if selection_strategy == 'conf':
                idx, c_idx = self.select_example_ByConf(id_conf, set(km_idx))
            if selection_strategy == 'disagree6':
                idx, c_idx = self.select_example_weak_KL_6(fix_answers, corpusObj, weakAnswers_dict)
            if selection_strategy == 'random':
                idx, c_idx = self.select_by_random()


            km_idx.append(idx)
            cl_id.append(c_idx)  # track picked cluster id on each iteration

            if label_propag:
                self.update_tao(km_idx, corpusObj)
            p_idx, p_label, p_dist, p_src = self.update_pseudo_set(idx, c_idx, p_idx, p_label, p_dist, p_src, corpusObj)

            # sub-cluster the cluster
            self.sub_cluster(c_idx, self.clf_al)

            if rr % iter_up_ans == 0:
                if flag_sub_cluster and averageH_hist[-1] > old_aveH * r_thresh:
                    self.revise_cluster(km_idx, p_idx, p_label)
                    # update true acc per cluster
                    self.updateTrueTransAcc_cluster(self.ex_id_cs)
                    self.m_transferAcc_true_hist.append((rr, self.m_accTransfer_cluster_true))

                    # compute entropy
                    averageH = self.computeAveClusterEntropy(self.ex_id_cs, corpusObj, self.clf)
                    averageH_hist.append(averageH)

                    old_aveH = averageH
                    clusterChange.append(rr)


                clusterNum_al.append((len(self.ex_id), len(self.ex_id_cs)))
                #print("cluster num ", clusterNum_al)

                if flag_update_agg:
                    # update agg
                    fix_answers = self.getfixAns(km_idx, p_idx, p_label)

                    weakAnswers_dict, weight, id_conf = self.IWMV_cluster(self.ex_id_cs, fix_answers)

                    weakAnswers_dict = {id: weakAnswers_dict[id] for id in weakAnswers_dict if
                                        id_conf[id] >= confid_thresh}

                    # oracle weight estimation
                    self.m_transferAcc_pred_hist.append((rr, weight))
                    rmse_cur = self.computeAccRMSE_cluster(self.m_accTransfer_cluster_true, weight)
                    self.m_rmsePredWeakAcc.append(rmse_cur)

            acc, trainNum, train_correct = self.get_pred_acc(fn_test, label_test, km_idx, p_idx, p_label,
                                                             weakAnswers_dict, corpusObj)
            self.m_accList.append(acc)

            # acc of clf trained on only on al
            acc_al, trainNum_al, train_correct_al = self.get_pred_acc(corpusObj.m_feature[self.m_test],
                                                                      corpusObj.m_label[self.m_test], km_idx, p_idx,
                                                                      p_label, dict(), corpusObj, self.clf_al)

            self.m_accList_al.append(acc_al)
            trainSampleNum.append(
                (rr, trainNum - trainNum_al, train_correct - train_correct_al, trainNum_al, train_correct_al))
            select_ins_info.append((rr, corpusObj.m_label[idx], responses[idx], trainNum - trainNum_al))

            # class entropy
            averageH = self.computeAveClusterEntropy(self.ex_id_cs, corpusObj, self.clf)
            averageH_hist.append(averageH)

            if rr % 5 == 0:
                print('iter: ',rr, ' acc: ',acc)

        print("finished!")

    def clusterWithFeatures(self, clusterNum, selectedID, features_cluster, seed):

        ex_id_cs = dd(list)  # example id for each C  # {clusterID: [idxs]}
        id2cluster = {}
        c_cs = KMeans(init='k-means++', n_clusters=clusterNum, n_init=10, random_state=seed)
        c_cs.fit(features_cluster)
        dist_cs = np.sort(c_cs.transform(features_cluster))  # size: instanceNum x clusterNum

        for i, j, k in zip(c_cs.labels_, selectedID, dist_cs):
            ex_id_cs[i].append(int(j))
            id2cluster[j] = i
        return ex_id_cs, id2cluster

    def statics_clusterRes(self, ex_id_cs, response):
        c_res = {}  # cid: response: num
        for cid in ex_id_cs:
            c_res_i = {}
            for id in ex_id_cs[cid]:
                res_i = str(response[id])
                if res_i not in c_res_i:
                    c_res_i[res_i] = 1
                else:
                    c_res_i[res_i] += 1
            c_res[cid] = c_res_i
        return c_res

def loadData(corpusObj, dataName):
    if dataName == "sensorTypes":
        target = args.target
        if target == 'rice':
            raw_pt = [i.strip().split('\\')[-1][:-5] for i in
                      open('../../dataset/sensorType/rice_pt_soda').readlines()]
            fn = get_name_features(raw_pt)

            featureMatrix = fn

            featureMatrix = np.array(featureMatrix)
            # labelArray = np.array(labelList)

            transferLabelFile = "../../dataset/sensorType/sdh_soda_rice_new/transferLabel_sdh--rice--RFC.txt"
            auditorLabelList, transferLabelList, trueLabelList = readTransferLabel(transferLabelFile)
            transferLabelArray = np.array(transferLabelList)

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

            auditorLabelArray = np.array(auditorLabelList6)
            labelArray = np.array(trueLabelList6)

            #
            transferLabelArrayList = []

            transferLabelArrayList.append(transferLabelArray)
            transferLabelArrayList.append(transferLabelArray2)
            transferLabelArrayList.append(transferLabelArray3)

            transferLabelArrayList.append(transferLabelArray4)

            transferLabelArrayList.append(transferLabelArray5)

            transferLabelArrayList.append(transferLabelArray6)

            multipleClassFlag = True
            initialExList = [[470, 352, 217], [203, 280, 54], [267, 16, 190], [130, 8, 318], [290, 96, 418],
                             [252, 447, 55],
                             [429, 243, 416], [240, 13, 68], [115, 449, 226], [262, 127, 381]]

            distinct_labels = list(set(trueLabelList6))

            corpusObj.initCorpus(featureMatrix, labelArray, transferLabelArrayList, auditorLabelArray, initialExList,
                                 "sensor",
                                 multipleClassFlag, distinct_labels, raw_pt)


def CVALParaWrapper(args):
    return CVALPerFold(*args)


def CVALPerFold(corpusObj, initialSampleList, gold_datast, weak_trn_dataset, train, test):
    StrongLabelNumThreshold = args.budget

    random.seed(10)
    np.random.seed(10)

    alObj = _ActiveClf(corpusObj.m_category, corpusObj.m_multipleClass, corpusObj.m_labelNum, StrongLabelNumThreshold)
    alObj.initActiveClf(initialSampleList, gold_datast, weak_trn_dataset, train, test)

    alObj.activeTrainClf_cluster_new(corpusObj)

    accList = alObj.m_accList

    resultPerFold = []
    resultPerFold.append(accList)

    return resultPerFold


def parallelCVAL(corpusObj, outputSrc, modelVersion):
    totalSampleNum = len(corpusObj.m_label)
    print("number of samples in dataset:", totalSampleNum)
    sampleIndexList = [i for i in range(totalSampleNum)]
    random.shuffle(sampleIndexList)

    # gold rate
    goldRate = 0.0  # 0.15
    goldRate2 = 1.0
    # weak oracle training dataset
    weakRate = 0.0  # 0.4

    foldNum = 10
    perFoldSampleNum = int(totalSampleNum * 1.0 / foldNum)
    foldSampleList = []

    for foldIndex in range(foldNum - 1):
        perFoldSampleList = sampleIndexList[foldIndex * perFoldSampleNum:(foldIndex + 1) * perFoldSampleNum]
        foldSampleList.append(perFoldSampleList)

    perFoldSampleList = sampleIndexList[perFoldSampleNum * (foldNum - 1):]
    foldSampleList.append(perFoldSampleList)

    totalAccList = [[] for i in range(foldNum)]

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

    ave_acc_supervised = sum(acc_supervised) / len(acc_supervised)
    print("supervised average acc on test: " + str(ave_acc_supervised))
    ave_acc_spv_trn = sum(acc_supervised_trn) / len(acc_supervised_trn)
    print("supervised average acc on train: " + str(ave_acc_spv_trn))
    # the main process with multiprocessing
    poolObj = Pool(poolNum)
    results = poolObj.map(CVALParaWrapper, argsList)
    poolObj.close()
    poolObj.join()

    for poolIndex in range(poolNum):
        foldIndex = poolIndex
        resultFold = results[foldIndex]
        totalAccList[foldIndex] = resultFold[0]

    postfix = args.postfix
    writeFile(outputSrc, modelVersion, totalAccList, "acc" + postfix)


if __name__ == '__main__':
    timeStart = datetime.now()

    corpusObj = _Corpus()
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
