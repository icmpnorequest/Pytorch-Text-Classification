# coding=utf-8
"""
@author: Yantong Lai
@date: 04/13/2020
@descrption: It aims to extract some data from the original data file.
"""

import pandas as pd
import os

dataset_path = "aclImdb"
original_train = os.path.join(dataset_path, "train.tsv")
original_test = os.path.join(dataset_path, "test.tsv")
# Number of train_data = 25000
# Number of test_data = 25000

new_train = os.path.join(dataset_path, "train_cpu.tsv")
new_test = os.path.join(dataset_path, "test_cpu.tsv")
new_valid = os.path.join(dataset_path, "valid_cpu.tsv")
# The ratio of train:valid:test is 8:1:1
# We wanna extract 800 rows in train.tsv, so 100 for valid and 100 for test

# The header in the tsv file
column_names = ['label', 'text']


def extract(orig_train, orig_test, new_train, new_valid, new_test, col_names):
    """It aims to extract some data from the old file."""
    # 1. Read original file
    df_orig_train = pd.read_csv(orig_train, sep="\t", names=col_names)
    df_orig_test = pd.read_csv(orig_test, sep="\t", names=col_names)

    # 2. Extract new train and valid file
    df_new_train = df_orig_train.iloc[:800]
    df_new_valid = df_orig_train.iloc[800:900]
    df_new_test = df_orig_test.iloc[:100]

    # 3. Save DataFrames to new tsv files
    df_new_train.to_csv(new_train, sep="\t", header=False, index=False)
    df_new_valid.to_csv(new_valid, sep="\t", header=False, index=False)
    df_new_test.to_csv(new_test, sep="\t", header=False, index=False)


if __name__ == '__main__':

    # Check if there exists new_train, new_valid and new_test
    if os.path.exists(new_train) and os.path.exists(new_valid) and os.path.join(new_test):
        print("There already exists new train, valid and test file.")

    else:
        extract(orig_train=original_train, orig_test=original_test, new_train=new_train,
                new_valid=new_valid, new_test=new_test, col_names=column_names)
