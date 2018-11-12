# encoding=utf8
"""
@author: yang
@contact: yang9112@gmail.com
@file: feature_honor.py
@time: 2018/11/12 9:23
"""
import os

from data.feature_tools import feature_tool
import pandas as pd


class feature_honor(feature_tool):
    def __init__(self, directory):
        self.features = pd.read_csv(os.path.join(directory, "企业表彰荣誉信息.csv"))

    def get_feature(self):
        return self.features.groupby(self.key_name).size().reset_index(name='honor_count')
