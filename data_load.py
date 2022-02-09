import pandas as pd 
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class DataProcessor():
  def __init__(self):
    self._cols_int = ['I{}'.format(idx) for idx in range(1,14)]
    self._single_cols_category = ['C{}'.format(idx) for idx in range(1,27)]
    # self._cols_int = ['I1','I2','I3']
    # self._single_cols_category = ['C1','C2','C3']
  
  def _build_cat_vocab(self,df,min_occur):
    self.tag2idx = []
    for i in range(len(self._single_cols_category)):
      self.tag2idx.append(defaultdict(int))
    
    for i,c in enumerate(self._single_cols_category):
      counts = df[c].value_counts()
      valid_counts = counts.loc[counts>=min_occur]
      
      for idx,tag in enumerate(valid_counts.index,start=1):
        self.tag2idx[i][tag] = idx
        
      self.tag2idx[i]['<unk>'] = 0
  
  
  def _normalize_numerics(self, data, upper_bound):
    numeric_features = data.loc[:, self._cols_int].copy()
    # axis=1,按列应用upper_bound,大于upper_bound，这将该值截断为upper_bound
    numeric_features.clip(upper=upper_bound, axis=1, inplace=True)

    # I2有小于0的值
    # -1    204968(占10%左右)
    # -2      1229
    # -3         1
    numeric_features['I2'] = (numeric_features['I2'] + 1).clip(lower=0)
    
    numeric_features = np.log1p(numeric_features)
    
    col_min = numeric_features.min()
    col_max = numeric_features.max()
    
    return (numeric_features - col_min) / (col_max - col_min)
  
  
  def _transform_categorical_row(self,row,*args):
    txts = []
    num = args[0]
    if(isinstance(num,str)):
      num = [num]
    
    for c in num:
      tag = row[c]
      if len(tag)==0:
        continue
      
      idx = self.tag2idx[self._single_cols_category.index(c)].get(tag,0)
      txts.append("{}".format(idx))
      
    return ",".join(txts)
  
  
  def _process(self,df):
    int_upper_bound=[20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
    
    df.fillna(value={c:'' for c in self._single_cols_category},inplace=True)
    
    dataset = pd.DataFrame()
    dataset['label'] = df['label']
    _X = df.loc[:,df.columns != 'label']
    
    normed_numeric_feats = self._normalize_numerics(_X,int_upper_bound)
    
    for colname in self._cols_int:
      dataset[colname] = normed_numeric_feats[colname]
      
    for _col_category in self._single_cols_category:
      dataset[_col_category] = _X.progress_apply(lambda row:self._transform_categorical_row(row,_col_category),
                                        axis=1)
    
    return dataset
      
      
if __name__ == "__main__":
  ratio = 0.2
  df = pd.read_csv("dataset\criteo_sampled_data.csv")
  train_df, test_df = train_test_split(df, test_size=ratio)
  
  tqdm.pandas()
  proc = DataProcessor()
  proc._build_cat_vocab(train_df,min_occur=20)
  
  train_set = proc._process(train_df)
  test_set = proc._process(test_df)
  
  out_dir = "./dataset"
  train_set.to_csv(out_dir+"/_train.csv", index=False, float_format='%.3f')
  test_set.to_csv(out_dir+"/_test.csv", index=False, float_format='%.3f')