# encoding=utf8
"""
@author: yang
@contact: yang9112@gmail.com
@file: pretreat.py
@time: 2018/11/10 20:43
"""
import os


def read_data():
    return


if __name__ == '__main__':
    train_file_path = r"C:\Users\admin\Desktop\竞赛相关\初赛\train"
    test_file_path = r"C:\Users\admin\Desktop\竞赛相关\初赛\test"
    test_list = []
    for file_name in os.listdir(test_file_path):
        test_list.append(file_name)

    for file_name in os.listdir(train_file_path):
        if file_name not in test_list:
            print(file_name)