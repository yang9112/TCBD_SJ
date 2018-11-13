# encoding=utf8
"""
@author: yang
@contact: yang9112@gmail.com
@file: label_cooker.py
@time: 2018/11/10 20:51
"""
import os

import pandas as pd
import xgboost as xgb
import numpy as np

from data.feature_abnormal import feature_abnormal
from data.feature_advert import feature_advert
from data.feature_honor import feature_honor

train_data_dir = r"C:\Users\admin\Desktop\竞赛相关\初赛\train"
test_data_dir = r"C:\Users\admin\Desktop\竞赛相关\初赛\test"
discredit_enterprise_files = ["双公示-法人行政处罚信息.csv", "失信被执行人名单.csv"]
col_x = ['honor_count', 'abnormal', 'advert']

train_features = [
    feature_honor(train_data_dir),
    feature_abnormal(train_data_dir),
    feature_advert(train_data_dir)
]

test_features = [
    feature_honor(test_data_dir),
    feature_abnormal(test_data_dir),
    feature_advert(test_data_dir)
]


# 双公示-法人行政许可信息
def double_publicity_feature(data_path):
    return pd.read_csv(os.path.join(data_path, "双公示-法人行政许可信息.csv")).drop_duplicates(subset=[key_name])


def get_x_index(data_path):
    pd_x = pd.read_csv(os.path.join(data_path, "企业基本信息&高管信息&投资信息.csv"))
    return pd.DataFrame(pd_x, columns=["企业名称"]).drop_duplicates(inplace=False)


# 双公示-法人行政处罚信息 & 失信被执行人名单
def get_y_label(x_index):
    pd_y_1 = pd.read_csv(os.path.join(train_data_dir, "双公示-法人行政处罚信息.csv"))
    pd_y_2 = pd.read_csv(os.path.join(train_data_dir, "失信被执行人名单.csv"))
    pd_y_1["FORTARGET1"] = 1
    pd_y_2["FORTARGET2"] = 1
    pd_y_label = pd.merge(pd_y_1, pd_y_2, on=key_name, how="outer")

    return pd.merge(x_index, pd_y_label, on=key_name, how="left").fillna(0)


def get_data(x_index, features):
    company_data = x_index
    for feature in features:
        company_data = pd.merge(company_data, feature.get_feature(), on=key_name, how="left")
    return company_data[col_x].fillna(0)


def train_and_predict(x_train, y_train, x_test):
    xg_train = xgb.DMatrix(x_train, y_train)
    clf = xgb.XGBClassifier()

    xgb_param = clf.get_xgb_params()
    xgb_param['silent'] = 0
    xgb_param['learning_rate'] = 0.1
    cv_result = xgb.cv(xgb_param, xg_train, num_boost_round=clf.get_params()['n_estimators'], nfold=5,
                       metrics='auc', early_stopping_rounds=10)
    clf.set_params(n_estimators=cv_result.shape[0])
    print(clf.get_params())

    clf.fit(x_train, y_train, eval_metric='auc')

    # 预测
    test_prob = clf.predict_proba(x_test)[:, 1]  # 1的概率
    return test_prob


if __name__ == '__main__':
    key_name = "企业名称"

    honor = feature_honor(train_data_dir)
    abnormal = feature_abnormal(train_data_dir)
    advert = feature_advert(train_data_dir)
    X_index = get_x_index(train_data_dir)

    X_train = get_data(X_index, train_features)
    y_label = get_y_label(X_index)

    X_index = get_x_index(test_data_dir)
    X_test = get_data(X_index, test_features)

    predict_prob_1 = train_and_predict(x_train=X_train[col_x], y_train=y_label["FORTARGET1"], x_test=X_test[col_x])
    predict_prob_2 = train_and_predict(x_train=X_train[col_x], y_train=y_label["FORTARGET2"], x_test=X_test[col_x])
    result = pd.DataFrame(data=np.array([predict_prob_1, predict_prob_2], dtype=np.float32).transpose(),
                          index=X_index[key_name], columns=["FORTARGET1", "FORTARGET2"])
    result.to_csv("../compliance_assessment.csv", index_label="EID")
