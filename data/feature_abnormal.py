# encoding=utf8
"""
@author: yang
@contact: yang9112@gmail.com
@file: feature_unnormal.py
@time: 2018/11/12 9:37
"""
import os

from data.feature_tools import feature_tool

import pandas as pd


class feature_abnormal(feature_tool):
    def __init__(self, directory):
        self.features = pd.read_csv(os.path.join(directory, "企业非正常户认定.csv"))

    def get_feature(self):
        return self.features.groupby(self.key_name).size().reset_index(name='abnormal')
