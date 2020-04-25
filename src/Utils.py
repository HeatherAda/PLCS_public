import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET

'''
load the data from dec theory
'''

def loadDecData(dataset,filepath):

    #util.main(dataset)
    mat, rel, orin2idx_instance = loadDec_Feat_Turk(dataset, filepath)  # mat: # tf-idf feature; rel: label 0/1

    responses, orin2idx_oracle = loadResponse(dataset, filepath)

    # responses (outside) -> (inside)
    responses = responseOut2In(responses, orin2idx_oracle, orin2idx_instance)

    #filter_labels = {0, 1}
    #matrix_res, oracle_idx, instance_idx, instance_feature =
    u_features = np.array([[1] for i in range(len(orin2idx_oracle))])
    u_features, v_features, responses = u_features, mat, responses

    label_idx = list(set([ri for ri in rel]))
    label_idx.sort()
    label_idx = {label_idx[i]: i for i in range(len(label_idx))}
    matrix_observed = response2matrix(responses, label_idx, len(orin2idx_oracle), len(orin2idx_instance))
    return u_features, v_features, responses, matrix_observed, label_idx, np.array(rel)


def loadDec_Feat_Turk(dataset,filepath):
    if dataset == 'proton-beam':
        pub_dic = get_pub_dic_xml(filepath)
        # pub_dic_items are already sorted by key
        [rec_nums, texts] = zip(*pub_dic.items())
        rel = get_relevant(filepath)
    else:
        pub_dic = get_pub_dic_csv(dataset,filepath)  # [Instance feature] {abstract_id:  title + abstract}
        # [rec_nums, texts] = zip(*pub_dic.items())
        (turk_dic, rel_dic) = get_turk_data(dataset)  # [Response] turk_dic: {abstract ID: [(Q3,Q4)]}; rel_dic: {abstract, label}
        texts = []
        all_ids = pub_dic.keys()
        #all_ids.sort() # sort by abstractID
        all_ids = sorted(all_ids)
        for i in all_ids:
            if i in pub_dic and i in turk_dic and i in rel_dic:
                texts.append(pub_dic[i])
            else:
                if i in pub_dic: pub_dic.pop(i)
                if i in turk_dic: turk_dic.pop(i)
                if i in rel_dic: rel_dic.pop(i)

        rel = [rel_dic[i] for i in all_ids if i in rel_dic]
        #(_, rel) = zip(*rel_dic.items())
        #rel = map(int, rel)
        rel = [int(i) for i in rel]

    vectorizer = TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1, 2), binary=True, max_features=500)
    #vectorizer = TfidfVectorizer()
    mat = vectorizer.fit_transform(texts)
    #return (pub_dic, texts)

    instances_filtered = pub_dic.keys()
    instances_filtered = sorted(instances_filtered)
    orin2idx_instance = {instances_filtered[i]: i for i in range(len(instances_filtered))}

    '''
    mat:  [tf-idf feature]
    rel:  {0: label}
    turk_dic: {abstract ID: [(Q3,Q4)]}
    '''
    return mat, rel, orin2idx_instance


def get_pub_dic_csv(dataset,filepath):
    filename = filepath + dataset + "-text.csv"
    f = open(filename)
    f.readline()
    csv_reader = csv.reader(f)

    # Create dic of : id -> feature text
    pub_dic = {}

    for row in csv_reader:
        (abstract_id, title, publisher, abstract) = tuple(row)[0:4]
        abstract_id = int(abstract_id)
        text = title + abstract

        pub_dic[abstract_id] = text

    return pub_dic

def loadResponse(dataset,filepath):
    responses = {}  # response: {instance: {oralce}}

    all_oracles = {}
    all_instances = {}

    filename = filepath + dataset + "-turk.csv"
    f = open(filename)
    first_line = f.readline()
    csv_reader = csv.reader(f)

    for row in csv_reader:
        # print len(row)
        if dataset == 'omega3':
            (AssignmentId, WorkerId, HITId, AcceptTime, SubmitTime, ApprovalTime, TimeToComplete, PMID, AbstractId,
             Question2, Question3, Question4, Relevant, Honeypot) = tuple(row)
        else:
            (AssignmentId, WorkerId, HITId, AcceptTime, SubmitTime, ApprovalTime, TimeToComplete, PMID, AbstractId,
             Question1, Question2, Question3, Question4, Relevant, Honeypot) = tuple(row)
        AbstractId = int(AbstractId)

        # oracles
        if WorkerId not in all_oracles: all_oracles[WorkerId] = 1
        else: all_oracles[WorkerId] += 1
        # instances
        if AbstractId not in all_instances: all_instances[AbstractId] = 1
        else: all_instances[AbstractId] += 1

        if AbstractId not in responses: responses[AbstractId] = {}
        responses[AbstractId][WorkerId] = get_answer(Question3, Question4, dataset)

    oracles_filtered = all_oracles.keys()
    oracles_filtered = sorted(oracles_filtered)
    orin2idx_oracle = {oracles_filtered[i]: i for i in range(len(oracles_filtered))}

    return responses, orin2idx_oracle


def get_turk_data(dataset):
    filename = "../../dataset/dec/" + dataset + "-turk.csv"
    f = open(filename)
    first_line = f.readline()
    csv_reader = csv.reader(f)

    turk_dic = {}
    rel_dic = {}
    for row in csv_reader:
        # print len(row)
        if dataset == 'omega3':
            (AssignmentId, WorkerId, HITId, AcceptTime, SubmitTime, ApprovalTime, TimeToComplete, PMID, AbstractId,
             Question2, Question3, Question4, Relevant, Honeypot) = tuple(row)
        else:
            (AssignmentId, WorkerId, HITId, AcceptTime, SubmitTime, ApprovalTime, TimeToComplete, PMID, AbstractId,
             Question1, Question2, Question3, Question4, Relevant, Honeypot) = tuple(row)
        AbstractId = int(AbstractId)
        if AbstractId not in turk_dic: turk_dic[AbstractId] = []
        turk_dic[AbstractId].append((Question3, Question4))
        rel_dic[AbstractId] = Relevant

    return (turk_dic, rel_dic)

def get_answer(q3, q4, dataset):
    NEG = 0
    UNOBSERVED = 0
    POS = 1

    if dataset == 'proton-beam':
        ans = POS
        if (q3 == 'No' or (not q4.isdigit())):
            ans = NEG
    else:
        ans = NEG
        if (q4 == 'CantTell' or q4 == 'Yes'):
            ans = POS
        #if (q4 == '-' and q3 == '-'):
        #    ans = UNOBSERVED

    return ans


def get_pub_dic_xml(filepath):
    file_name = filepath + 'proton-beam-all.xml'

    tree = ET.parse(file_name)
    root = tree.getroot()[0]

    # Create dic of : id -> feature text
    pub_dic = {}
    for pub in root:
        rec_number = int(get_text(pub.find('rec-number')))
        abstract = get_text(pub.find('abstract'))
        title = get_text(pub.find('titles')[0])
        text = title + abstract
        for kw in pub.find('keywords'):
            text = text + kw.text + ' '
        pub_dic[rec_number] = text

    return pub_dic

def get_text(a):
    try:
        return a.text
    except AttributeError:
        return ''

def get_relevant(filepath):
    f = open(filepath+'proton-beam-relevant.txt')
    res = np.zeros(4751)
    for line in f:
        x = int(line)
        res[x-1] = 1
    f.close()

def responseOut2In(responses, oracle_idx, instance_idx):
    inside_response = {instance_idx[abstractID]:{oracle_idx[oid]: responses[abstractID][oid] for oid in responses[abstractID]} for abstractID in responses}
    return inside_response

def response2matrix(responses_inside, label_idx, oracle_num, instance_num):
    matrix= np.zeros([len(label_idx), oracle_num,
                      instance_num])  # 0 or 1 [lb(0,1,2,..): [oi(0,1,2,..):[instance(0,1,2,..)]]]

    for id in responses_inside:
        for oi in responses_inside[id]:
            rs = responses_inside[id][oi]
            li = label_idx[rs]
            matrix[li][oi][id] = 1.0
    return matrix

'''
select one oracle
input:
    oracleid: (inner)
    response: (inner)
'''
def selectInstanceByOracle(responses, v_features, truelabel, oracleid):
    select_instanceid = []
    for id in responses:
        if oracleid in responses[id]:
            select_instanceid.append(id)
    select_instanceid.sort()
    select_label = truelabel[select_instanceid]
    select_noisylabel = [responses[id][oracleid] for id in select_instanceid]
    select_features = v_features[select_instanceid]
    return select_instanceid, select_features, select_label, select_noisylabel

'''
return the oracleid which answers most questions
input: 
    response: inner
'''
def max_oracle(responses, oracleNum):
    num_answer = [0 for i in range(oracleNum)]
    for id in responses:
        for oi in responses[id]:
            num_answer[oi] += 1
    max_oi = num_answer.index(max(num_answer))
    return max_oi

def noisylabelQuality(true_labels, noisylabel):
    classes = list(set(true_labels))
    correct = {lb: 0 for lb in classes}
    label_num = {lb: (true_labels == lb).sum() for lb in classes}
    for i in range(len(true_labels)):
        if true_labels[i] == noisylabel[i]:
            correct[true_labels[i]] += 1
    acc_per_class = {lb: float(correct[lb])/label_num[lb] for lb in correct}
    return acc_per_class, label_num
            

'''
    responses: dict{instance: {oracle: annotation}}
'''
if __name__ == '__main__':
    dataName = "appendicitis" #appendicitis
    filepath = "../../dataset/dec/"
    u_features, v_features, responses, matrix_sparse, filter_labels, true_labels = loadDecData(dataName,filepath)
    print("oracle num: ", u_features.shape[0], "instance num:", v_features.shape[0])

    # find max oi
    max_oi = max_oracle(responses, u_features.shape[0])  # the oracle which answers most number of questions
    select_instanceid, select_features, select_label, select_noisylabel = \
        selectInstanceByOracle(responses, v_features, true_labels, max_oi)
    acc_per_class, label_num = noisylabelQuality(select_label, select_noisylabel)
    print("size: ", len(select_instanceid), "oracleid: ", max_oi, "label distri: ", label_num, "oracle quality: ", acc_per_class)


