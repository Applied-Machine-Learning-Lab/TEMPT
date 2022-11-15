from xml.dom.pulldom import ErrorHandler
import pandas as pd
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import BRICS
import pickle
import argparse

##### process medications #####
# load patient data
def pat_process(pat_file, hos_file):

    pat_pd = pd.read_csv(pat_file)
    pat_pd = pat_pd[['patientunitstayid', 'patienthealthsystemstayid', 
                     'hospitaladmitoffset', 'hospitalid',]]
    # 这里的startdate是为了给同一个subject下的不同hadm排序
    pat_pd.rename(columns={'patientunitstayid': 'hadm_id', 
                   'patienthealthsystemstayid': 'subject_id',
                   'hospitaladmitoffset': 'hadm_startdate',
                   'hospitalid': 'hospital_id',}, inplace=True)
    hos_pd = pat_pd[['hadm_id', 'hospital_id']]
    pickle.dump(hos_pd, file=open(hos_file, 'wb'))

    return pat_pd


# load med data
def med_process(med_file, pat_pd):

    med_pd = pd.read_csv(med_file)

    med_pd = med_pd[['patientunitstayid', 'drugstartoffset', 'drughiclseqno']]
    med_pd.rename(columns={'patientunitstayid': 'hadm_id', 
                           'drugstartoffset': 'med_startdate',
                           'drughiclseqno': 'drug_id'}, inplace=True)
    med_pd.drop(med_pd[med_pd.drug_id.isna()].index, inplace=True)   # remove no HICL medicine
    med_pd.drop_duplicates(inplace=True)    # drop the duplications 
    med_pd = pd.merge(med_pd, pat_pd, on='hadm_id', how='left')
    # 先排序每个icu unit记录，再排序用药记录
    med_pd.sort_values(by=['subject_id', 'hadm_startdate', 'med_startdate'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    return med_pd


##### process diagnosis #####
def diag_process(diag_file, pat_pd):
    diag_pd = pd.read_csv(diag_file)
    diag_pd = diag_pd[['patientunitstayid', 'diagnosisoffset', 'icd9code']]
    diag_pd.rename(columns={'patientunitstayid': 'hadm_id', 'diagnosisoffset': 'diag_startdate', 'icd9code': 'icd9_code'}, inplace=True)
    diag_pd.dropna(inplace=True)
    #diag_pd.drop(columns=['seq_num','row_id'],inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd = pd.merge(diag_pd, pat_pd, on='hadm_id', how='left')
    # 先排序每个icu unit记录，再排序用药记录
    diag_pd.sort_values(by=['subject_id', 'hadm_startdate', 'diag_startdate'], inplace=True)
    #diag_pd.sort_values(by=['subject_id','hadm_id'], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    def filter_2000_most_diag(diag_pd):
        diag_count = diag_pd.groupby(by=['icd9_code']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
        diag_pd = diag_pd[diag_pd['icd9_code'].isin(diag_count.loc[:1999, 'icd9_code'])]
        
        return diag_pd.reset_index(drop=True)

    diag_pd = filter_2000_most_diag(diag_pd)

    return diag_pd

##### process procedure #####
def procedure_process(procedure_file, pat_pd):
    pro_pd = pd.read_csv(procedure_file, dtype={'icd9_code':'category'})
    pro_pd = pro_pd[['patientunitstayid', 'treatmentoffset', 'treatmentstring']]
    pro_pd.rename(columns={'patientunitstayid': 'hadm_id', 'treatmentoffset': 'treat_startdate', 'treatmentstring': 'icd9_code'}, inplace=True)
    #pro_pd.drop(columns=['row_id'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd = pd.merge(pro_pd, pat_pd, on='hadm_id', how='left')
    pro_pd.sort_values(by=['subject_id', 'hadm_startdate', 'treat_startdate'], inplace=True)
    #pro_pd.drop(columns=['seq_num'], inplace=True)
    #pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd


# visit >= 2, delete the single visit, and convert to sequence in hadm_id
def process_visit_lg2(med_pd):
    a = med_pd[['subject_id', 'hadm_id']].groupby(by='subject_id')['hadm_id'].unique().reset_index()
    a['hadm_id_len'] = a['hadm_id'].map(lambda x:len(x))
    a = a[a['hadm_id_len'] > 1]
    return a 


##---Filter out unpopular medication/diagnosis/procedure---##
def filter_pro(pro_pd, num):

    print('filter procedures')
    pro_count = pro_pd.groupby(by=['icd9_code']).size().reset_index().rename(
        columns={0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['icd9_code'].isin(
        pro_count.loc[:num, 'icd9_code'])]

    return pro_pd.reset_index(drop=True)


def filter_diag(diag_pd, num=128):  # 筛选出最热门的前num种diagnosis，剩下删去

    print('filter diagnosis')
    diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(
        columns={0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:num, 'ICD9_CODE'])]

    return diag_pd.reset_index(drop=True)


def filter_med(med_pd, num=299):

    print('filter medications')
    med_count = med_pd.groupby(by=['drug_id']).size().reset_index().rename(columns={
        0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
    med_pd = med_pd[med_pd['drug_id'].isin(med_count.loc[:num, 'drug_id'])]

    return med_pd.reset_index(drop=True)
##--- End ---##


###### combine three tables #####
def combine_process(med_pd, diag_pd, pro_pd):

    med_pd_key = med_pd[['subject_id', 'hadm_id']].drop_duplicates()
    diag_pd_key = diag_pd[['subject_id', 'hadm_id']].drop_duplicates()
    pro_pd_key = pro_pd[['subject_id', 'hadm_id']].drop_duplicates()
    # get the unique key (exits in three pd simultaneously)
    combined_key = med_pd_key.merge(diag_pd_key, on=['subject_id', 'hadm_id'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['subject_id', 'hadm_id'], how='inner')
    # delete the sample out of combined key
    diag_pd = diag_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['subject_id','hadm_id'])['icd9_code'].unique().reset_index()  # use unique() to get the sequence
    med_pd = med_pd.groupby(by=['subject_id', 'hadm_id'])['drug_id'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['subject_id','hadm_id'])['icd9_code'].unique().reset_index().rename(columns={'icd9_code':'pro_code'})  
    med_pd['drug_id'] = med_pd['drug_id'].map(lambda x: list(x))  # convert array to list
    pro_pd['pro_code'] = pro_pd['pro_code'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['subject_id', 'hadm_id'], how='inner')
    data = data.merge(pro_pd, on=['subject_id', 'hadm_id'], how='inner')
    #     data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data['drug_id_num'] = data['drug_id'].map(lambda x: len(x))

    return data

def statistics(data):
    print('#patients ', data['subject_id'].unique().shape)
    print('#clinical events ', len(data))
    
    diag = data['icd9_code'].values
    med = data['drug_id'].values
    pro = data['pro_code'].values
    
    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])
    
    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))
    print('#procedure', len(unique_pro))
    
    avg_diag, avg_med, avg_pro, max_diag, max_med, max_pro, cnt, max_visit, avg_visit = [0 for i in range(9)]

    for subject_id in data['subject_id'].unique():
        item_data = data[data['subject_id'] == subject_id]
        x, y, z = [], [], []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['icd9_code']))
            y.extend(list(row['drug_id']))
            z.extend(list(row['pro_code']))
        x, y, z = set(x), set(y), set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y) 
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt
        
    print('#avg of diagnoses ', avg_diag/ cnt)
    print('#avg of medicines ', avg_med/ cnt)
    print('#avg of procedures ', avg_pro/ cnt)
    print('#avg of vists ', avg_visit/ len(data['subject_id'].unique()))
    
    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of procedures ', max_pro)
    print('#max of visit ', max_visit)


##### indexing file and final record
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


# create voc set
def create_str_token_mapping(df, vocabulary_file):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()
    
    for index, row in df.iterrows():
        diag_voc.add_sentence(row['icd9_code'])
        med_voc.add_sentence(row['drug_id'])
        pro_voc.add_sentence(row['pro_code'])
    
    pickle.dump(obj={'diag_voc':diag_voc, 'med_voc':med_voc ,'pro_voc':pro_voc}, file=open(vocabulary_file,'wb'))
    return diag_voc, med_voc, pro_voc


# create final records
def create_patient_record(df, diag_voc, med_voc, pro_voc, record_file):
    records = [] # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df['subject_id'].unique():
        item_df = df[df['subject_id'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row['icd9_code']])
            admission.append([pro_voc.word2idx[i] for i in row['pro_code']])
            admission.append([med_voc.word2idx[i] for i in row['drug_id']])
            patient.append(admission)
        records.append(patient) 
    pickle.dump(obj=records, file=open(record_file, 'wb'))
    return records



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="demo", type=str, choices=['raw', 'demo'], help="Preprocess the raw or demo data of eICU.")
    args = parser.parse_args()

    output_dir = './data/eicu/handled/'
    dataset = args.dataset # 'demo or raw

    # files can be downloaded from https://eicu-crd.mit.edu/gettingstarted/access/
    # Raw data path
    med_file = './data/eicu/' + dataset + '/medication.csv'
    diag_file = './data/eicu/' + dataset + '/diagnosis.csv'
    procedure_file = './data/eicu/' + dataset + '/treatment.csv'
    patient_file = './data/eicu/' + dataset + '/patient.csv'

    # Processed data path
    if dataset == 'demo':
        record_file = output_dir + 'data-single-visit.demo.pkl'
        vocabulary_file = output_dir + 'vocab.demo.pkl'
        hospital_record = output_dir + 'hospital-record.demo.pkl'
    elif dataset == 'raw':
        record_file = output_dir + 'data-single-visit.raw.pkl'
        vocabulary_file = output_dir + 'vocab.raw.pkl'
        hospital_record = output_dir + 'hospital-record.raw.pkl'
    else:
        raise ValueError('Please select data from "raw" and "demo".')
    
    # for patient
    pat_pd = pat_process(patient_file, hospital_record)

    # for med
    med_pd = med_process(med_file, pat_pd)
    #med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)    
    #med_pd = med_pd.merge(med_pd_lg2[['subject_id']], on='subject_id', how='inner').reset_index(drop=True) # filter out the single visit

    med_pd = filter_med(med_pd, 300)    # filter out some uncommon medications
    print ('complete medication processing')

    # for diagnosis
    diag_pd = diag_process(diag_file, pat_pd)

    print ('complete diagnosis processing')

    # for procedure
    pro_pd = procedure_process(procedure_file, pat_pd)
    pro_pd = filter_pro(pro_pd, 1000)

    print ('complete procedure processing')

    # combine
    data = combine_process(med_pd, diag_pd, pro_pd)
    print ('complete combining\n')
    print ('The statistics of processed dataset:\n')
    statistics(data)

    # create vocab: the voc save the dict of new index and old index
    diag_voc, med_voc, pro_voc = create_str_token_mapping(data, vocabulary_file)
    print ("obtain voc")

    # create ehr sequence data
    #records = create_patient_record(data, diag_voc, med_voc, pro_voc, record_file)
    pickle.dump(obj=data, file=open(record_file, 'wb'))
    print ("obtain single-visit data")

