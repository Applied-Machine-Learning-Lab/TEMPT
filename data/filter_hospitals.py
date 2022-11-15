# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os

# %%
hos_id = pickle.load(open('./eicu/handled/hospital-record.raw.pkl', 'rb'))
data = pickle.load(open('./eicu/handled/data-single-visit.raw.pkl', 'rb'))

# %%
data = pd.merge(data, hos_id, how='left', on='hadm_id')

# %%
# 验证是否所有记录都有hospital id
data.hospital_id.isna().sum()

# %%
temp = data.groupby('hospital_id')['hadm_id'].agg('count')
filter_hos = temp[temp>=400].index

# %%
print('Before filtering: %d samples' % data.shape[0])
data_filter = data.loc[data.hospital_id.isin(filter_hos)]
print('After filtering: %d samples' % data_filter.shape[0])

# %%
pat = pd.read_csv('./eicu/raw/patient.csv')
pat = pat[['patientunitstayid', 'unitdischargelocation']]
pat.rename(columns={'patientunitstayid': 'hadm_id',
                    'unitdischargelocation': 'attri'}, inplace=True)
attri_dict = dict(zip(pat.attri.unique(), range(pat.attri.nunique())))
pat['attri'] = pat['attri'].map(attri_dict)
pat.drop(pat[pat.attri.isnull()].index, inplace=True)
pat['attri'] = pat['attri'].astype(int)

# %%
data_filter = pd.merge(data_filter, pat, on='hadm_id', how='left')
data_filter.drop(data_filter[data_filter.attri.isnull()].index, inplace=True)
print('After merge: %d samples' % data_filter.shape[0])

# %%
def split_dataset(data, seed):
        '''Split the datatset based on the ratio 8:1:1'''
        np.random.seed(seed)
        index_list = list(range(data.shape[0]))
        train_num, val_num = int(0.8 * data.shape[0]), int(0.1 * data.shape[0])

        train_index = np.random.choice(index_list, size=train_num, replace=False)
        val_index = np.random.choice(list(set(index_list) - set(train_index)), size=val_num, replace=False)
        test_index = list(set(index_list) - set(train_index) - set(val_index))

        return [data.iloc[train_index, :], data.iloc[val_index, :], data.iloc[test_index, :]]

# %%
data_multi_center = {}

for id in tqdm(filter_hos.values):

    data_multi_center[id] = data_filter.loc[data_filter.hospital_id==id]

# %%
for hos_id in tqdm(data_multi_center.keys()):
    
    for seed in [42, 43, 44, 45, 46]:
        
        folder_path = './eicu/handled/' + str(seed) + '/' + str(hos_id) + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        hos_df = data_multi_center[hos_id]
        train_df, val_df, test_df = split_dataset(hos_df, seed)
        pickle.dump(train_df, open(folder_path+'train.pkl', 'wb'))
        pickle.dump(val_df, open(folder_path+'val.pkl', 'wb'))
        pickle.dump(test_df, open(folder_path+'test.pkl', 'wb'))

# %%
with open('./eicu/handled/data-single-visit-multi-center.attri.raw.pkl', 'wb') as f:
    pickle.dump(data_multi_center, f)

# %%
filter_hos.values

# %%
temp = data.groupby('hospital_id')['hadm_id'].agg('count')
temp = temp[temp>=400]

hos_list = pd.DataFrame({
    'hosipital_id': temp.index,
    'record_num': temp.values
})
hos_list.to_csv('./eicu/handled/hospital.csv', index=False)


