import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV


np.set_printoptions(suppress=True)          #解除科学计数法显示
data = pd.read_csv("E:\GBDT+LR\dataset\demo.csv",float_precision='%.3f')
train_x = np.array(data.iloc[:, [i for i in range(1,data.shape[1])]])
train_y = np.array(data['label'])

params = {
  'boosting_type': 'gbdt',
  'objective': 'binary',
  'metric': 'auc',
  'max_depth': 5,
  'num_leaves': 30,
  'subsample':0.8,           # 数据采样
  'colsample_bytree': 0.8,   # 特征采样
  'learning_rate': 0.1,
  'n_estimators': 50,
}

params_test1={'max_depth': range(3,8,1)}
              
gsearch1 = GridSearchCV(estimator = lgb.LGBMClassifier(**params), return_train_score=True, 
                       param_grid = params_test1, scoring=['roc_auc','neg_log_loss'],refit='roc_auc',cv=5,n_jobs=-1)
fit_params = {
  'feature_name' : ["I1","I2","I3","I4","I5","I6","I7","I8","I9","I10","I11","I12","I13","C24","C25","C26"],
  'categorical_feature' : ["C24","C25","C26"],
}
gsearch1.fit(train_x,train_y,**fit_params)
print(gsearch1.cv_results_)
print('=======================')
print(gsearch1.best_params_, gsearch1.best_score_)

#可以把所有要search的parameter写在一起

# X, y = load_diabetes(return_X_y=True)
# reg = xgb.XGBRegressor(
#     tree_method="hist",
#     eval_metric=mean_absolute_error,
# )
# reg.fit(X, y, eval_set=[(X, y)])