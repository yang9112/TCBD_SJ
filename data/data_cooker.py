# encoding=utf8
"""
@author: yang
@contact: yang9112@gmail.com
@file: label_cooker.py
@time: 2018/11/10 20:51
"""
import os

import pandas as pd

train_data_dir = r"C:\Users\admin\Desktop\竞赛相关\初赛\train"
discredit_enterprise_files = ["双公示-法人行政处罚信息.csv", "失信被执行人名单.csv"]


# 双公示-法人行政许可信息
def double_publicity_feature(data_path):
    return pd.read_csv(os.path.join(data_path, "双公示-法人行政许可信息.csv"))


def get_x_index(data_path):
    pd_x = pd.read_csv(os.path.join(data_path, "企业基本信息&高管信息&投资信息.csv"))
    return pd.DataFrame(pd_x, columns=["企业名称"]).drop_duplicates()


# 双公示-法人行政处罚信息 & 失信被执行人名单
def get_y_label():
    pd_y_1 = pd.read_csv(os.path.join(train_data_dir, "双公示-法人行政处罚信息.csv"))
    pd_y_2 = pd.read_csv(os.path.join(train_data_dir, "失信被执行人名单.csv"))
    pd_y_1["FORTARGET1"] = '1'
    pd_y_2["FORTARGET2"] = '1'
    pd_y_label = pd.merge(pd_y_1, pd_y_2, on=key_name, how="outer")
    return pd_y_label.where(pd_y_label.notnull(), 0)


def get_data(x_index, y, features):
    # todo
    company_data = pd.merge(x_index, y, on=key_name, how="left")
    return company_data.where(company_data.notnull(), 0)


if __name__ == '__main__':
    key_name = "企业名称"
    y_label = get_y_label()
    # df_labels = pd.merge(x_index, df_label_y, how="left", on=["企业名称"])
    # print(df_labels.where(df_labels.notnull(), 0))
    print(get_data(get_x_index(train_data_dir), y_label, 1))
