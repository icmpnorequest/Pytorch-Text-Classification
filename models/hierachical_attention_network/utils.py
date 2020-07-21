# coding=utf-8
"""
@author: Yantong Lai
@date: 07/20/2020
@description: Utils of data file
"""

import os
import csv
import re

import pandas as pd
import numpy as np
import torch
from nltk.tokenize import sent_tokenize, word_tokenize

# dataset path and original csv file
agnews_dataset_path = "../../agnews/"
agnews_orig_train_file = "../../agnews/train.csv"
agnews_orig_test_file = "../../agnews/test.csv"

# processed csv file
agnews_processed_train_csv = "../../agnews/processed_train.csv"
agnews_processed_val_csv = "../../agnews/processed_val.csv"
agnews_processed_test_csv = "../../agnews/processed_test.csv"

# processed ndarray
processed_train_docs_file = "../../agnews/processed_train_docs.npy"
processed_val_docs_file = "../../agnews/processed_val_docs.npy"
processed_test_docs_file = "../../agnews/processed_test_docs.npy"
processed_train_labels_file = "../../agnews/processed_train_labels.npy"
processed_val_labels_file = "../../agnews/processed_val_labels.npy"
processed_test_labels_file = "../../agnews/processed_test_labels.npy"

# embed file
embed_file_path = "../../glove/glove.6B.100d.txt"

# max sentences number and words number
MAX_SENTS_NUM = 2
MAX_WORDS_NUM = 30


class AgNewsDataHelper:
    def __init__(self, dataset_path, train_file, test_file):
        self.dataset_path = dataset_path
        self.train_file = train_file
        self.test_file = test_file

    def get_labels_docs(self):
        """
        This method aims to load AgNews raw csv file.
        AgNews csv file contains 3 columns: label, title and doc
        return: <np.ndarray> train_labels, train_docs, test_labels, test_docs
        """
        df_train = pd.read_csv(self.train_file, names=['label', 'title', 'doc'])
        df_test = pd.read_csv(self.test_file, names=['label', 'title', 'doc'])
        train_labels = df_train['label'].values
        train_docs = df_train['doc'].values
        test_labels = df_test['label'].values
        test_docs = df_test['doc'].values
        return train_labels, train_docs, test_labels, test_docs

    def preprocess_docs(self, docs_array):
        """
        This method aims to pre-process docs array, e.g., remote stop words.
        """
        # Remove stopwords
        new_docs_list = []
        for doc in docs_array:
            temp = doc.lower()
            if "\\" in temp:
                temp = temp.replace("\\", "")
            if "&lt;FONT face" in temp:
                temp = re.sub(r"&lt;FONT face.*$", "", doc)
            if "a href=" in temp:
                temp = re.sub(r" ;a href.*/a&gt;", "", doc)
            if "A HREF" in temp:
                temp = re.sub(r" ;A HREF.*/A&gt;", "", doc)
            if "&lt" in temp:
                temp = temp.replace("&lt", "")
            if "i&gt" in temp:
                temp = temp.replace("i&gt", "")
            if "    " in temp:
                temp = temp.replace("    ", " ")
            if "#151" in temp:
                temp = temp.replace("#151", "")
            if "#36" in temp:
                temp = temp.replace("#36", "")
            if "#39" in temp:
                temp = temp.replace("#39", "")
            new_docs_list.append(temp)
        return np.array(new_docs_list)

    def reformat_label_values(self, labels_array):
        """
        This method aims to reformat labels array.
        Original label: 1, 2, 3, 4
        Output: 0, 1, 2, 3
        """
        if isinstance(labels_array, np.ndarray):
            reformat_labels_array = labels_array - 1
            return reformat_labels_array
        else:
            print("type({}) is not np.ndarray".format(labels_array))
            return labels_array

    def count_words_sents(self, doc_array):
        """
        This method aims to count number of sentences and number of words in a sentence.
        """
        total_num_sents = []
        total_num_words = []
        for doc in doc_array:
            sents = sent_tokenize(doc)
            total_num_sents.append(len(sents))
            temp_num_words = []
            for sent in sents:
                num_words = word_tokenize(sent)
                temp_num_words.append(len(num_words))
            total_num_words.append(temp_num_words)
        return np.array(total_num_sents), np.array(total_num_words)

    def split_train_val(self, train_docs_array, train_labels_array):
        """
        This method aims to split train array into train array and val array.
        """
        val_docs_array = train_docs_array[:7600]    # len of test.csv is 7600
        val_labels_array = train_labels_array[:7600]
        train_docs_array = train_docs_array[7600:]
        train_labels_array = train_labels_array[7600:]
        return train_docs_array, train_labels_array, val_docs_array, val_labels_array


class EmbedDataHelper:
    def __init__(self, embed_file):
        self.embed_file = embed_file

    def load_vocab(self):
        """
        This method aims to load vocab from embed file
        :return: <list> keys, <list> values
        """
        keys = []
        values = []
        with open(self.embed_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            key = line.split(" ")[0]
            value = line.split(" ")[1:]
            keys.append(key)
            values.append(value)
        # form <dict>
        # vocab = dict(zip(keys, values))
        return keys, values

    def get_vocab_size(self, keys):
        return len(keys) + 1

    def generate_keys_dict(self, keys):
        keys_dict = dict(zip(keys, range(len(keys))))
        return keys_dict

    def get_oov_index(self, keys):
        return len(keys)


def document_to_index(doc, keys_dict):
    sents = sent_tokenize(doc)
    oov_index = len(keys_dict)
    doc_index = []
    for sent in sents:
        words = word_tokenize(sent)
        words_index = []
        for word in words:
            if word in keys_dict.keys():
                words_index.append(keys_dict[word])
            else:
                words_index.append(oov_index)
        doc_index.append(words_index)
    return doc_index


def document_index_to_feature(doc_index, oov_index, fix_sent_num, fix_word_num):
    """
    This method aims to generate feature array with the shape of (fix sent num, fix word num)
    :return: <list> feature
    """
    # Generate feature
    feature = []
    for _ in range(fix_sent_num):
        feature.append([oov_index] * fix_word_num)

    if len(doc_index) == 1:
        if len(doc_index[0]) >= fix_word_num:
            feature[0] = doc_index[0][:fix_word_num]
        else:
            feature[0][:len(doc_index[0])] = doc_index[0]
    else:
        for idx in range(fix_sent_num):
            if len(doc_index[idx]) >= fix_word_num:
                feature[idx] = doc_index[idx][:fix_word_num]
            else:
                feature[idx][:len(doc_index[idx])] = doc_index[idx]
    return feature


def save_ndarray(ndarray, array_filename):
    with open(array_filename, 'wb') as f:
        np.save(f, ndarray)


def load_ndarray(array_filename):
    with open(array_filename, 'rb') as f:
        ndarray = np.load(f)
    return ndarray


def save_ndarray_to_csv(docs_array, labels_array, csv_file):
    """
    This method aims to save ndarray to csv file.
    """
    processed_array = np.vstack([labels_array, docs_array]).T
    df = pd.DataFrame(data=processed_array)
    df.to_csv(csv_file, index=False, header=['label', 'text'])


def transform_array_with_glove(docs_array, keys_dict, oov_index, fix_sent_num, fix_word_num):
    """
    This method aims to transform docs array according to glove.
    """
    temp_features = []
    for doc_idx in range(docs_array.shape[0]):
        doc_index = document_to_index(docs_array[doc_idx], keys_dict)
        feature = document_index_to_feature(doc_index, oov_index, fix_sent_num=fix_sent_num,
                                            fix_word_num=fix_word_num)
        temp_features.append(feature)
    temp_array = np.array(temp_features)
    return temp_array


if __name__ == '__main__':

    # 1. AgNewsDataHelper
    # (1) Create an AgNewsDataHelper instance
    AgNewsdatahelper = AgNewsDataHelper(dataset_path=agnews_dataset_path, train_file=agnews_orig_train_file,
                                        test_file=agnews_orig_test_file)

    # (2) Get labels and docs from original csv file
    train_labels_array, train_docs_array, test_labels_array, test_docs_array = AgNewsdatahelper.get_labels_docs()

    # (3) Process docs array
    train_docs_array = AgNewsdatahelper.preprocess_docs(train_docs_array)
    test_docs_array = AgNewsdatahelper.preprocess_docs(test_docs_array)

    # (4) Reformat labels_array
    train_labels_array = AgNewsdatahelper.reformat_label_values(labels_array=train_labels_array)
    test_labels_array = AgNewsdatahelper.reformat_label_values(labels_array=test_labels_array)

    # (5) Split train array into train array and val array
    train_docs_array, train_labels_array, val_docs_array, val_labels_array = AgNewsdatahelper.split_train_val(
        train_docs_array=train_docs_array,
        train_labels_array=train_labels_array)

    # (6) Save array to csv file
    save_ndarray_to_csv(docs_array=train_docs_array, labels_array=train_labels_array,
                        csv_file=agnews_processed_train_csv)
    save_ndarray_to_csv(docs_array=val_docs_array, labels_array=val_labels_array,
                        csv_file=agnews_processed_val_csv)
    save_ndarray_to_csv(docs_array=test_docs_array, labels_array=test_labels_array,
                        csv_file=agnews_processed_test_csv)

    # 2. EmbedDataHelper
    # (1) Create an EmbedDataHelper instance
    embedDataHelper = EmbedDataHelper(embed_file=embed_file_path)

    # (2) Get keys and values from glove
    keys, values = embedDataHelper.load_vocab()
    keys_dict = embedDataHelper.generate_keys_dict(keys)
    oov_index = embedDataHelper.get_oov_index(keys)

    # (3) Transform train, val and test's docs array
    processed_train_docs_array = transform_array_with_glove(docs_array=train_docs_array, keys_dict=keys_dict,
                                                            oov_index=oov_index, fix_sent_num=MAX_SENTS_NUM,
                                                            fix_word_num=MAX_WORDS_NUM)
    processed_val_docs_array = transform_array_with_glove(docs_array=val_docs_array, keys_dict=keys_dict,
                                                          oov_index=oov_index, fix_sent_num=MAX_SENTS_NUM,
                                                          fix_word_num=MAX_WORDS_NUM)
    processed_test_docs_array = transform_array_with_glove(docs_array=test_docs_array, keys_dict=keys_dict,
                                                           oov_index=oov_index, fix_sent_num=MAX_SENTS_NUM,
                                                           fix_word_num=MAX_WORDS_NUM)

    # (4) Save ndarray to file
    save_ndarray(ndarray=processed_train_docs_array, array_filename=processed_train_docs_file)
    save_ndarray(ndarray=processed_val_docs_array, array_filename=processed_val_docs_file)
    save_ndarray(ndarray=processed_test_docs_array, array_filename=processed_test_docs_file)
    save_ndarray(ndarray=train_labels_array, array_filename=processed_train_labels_file)
    save_ndarray(ndarray=val_labels_array, array_filename=processed_val_labels_file)
    save_ndarray(ndarray=test_labels_array, array_filename=processed_test_labels_file)
    print("Save arrays to files successfully.\n")

