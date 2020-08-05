"""
data loader for venue + onemode network

"""
import sys

sys.path.append('./')

import logging
from pathlib import Path
import string
import numpy as np
import random
import pickle as pkl
import csv
from scipy import sparse
from xlrd import open_workbook
# import tensorflow as tf
from src_gat_fusion2.utils.config import get_config_from_json, update_config_by_summary, update_config_by_datasize
from src_gat_fusion2.utils.dirs import create_dirs
from src_gat_fusion2.utils.logger import Logger
from src_gat_fusion2.utils.utils import get_args, load_adj_csv, load_adj_excel, adj_to_bias, \
    sparse_to_tuple, normalize_adj

random.seed(2010)

REDUCE_GRAPH_FEATURES = True

# flexibly define attributes and select for experiments
selected_attributes = [
                       'race',
                       'age_w1',
                       'black',
                       'hispanicity',
                       'smallnet_w1',
                       'education_w1',
                       'sexual_identity_w1',
                       'past12m_homeless_w1',
                       'insurance_w1',
                       'inconsistent_condom_w1',
                       'ever_jailed_w1',
                       'age_jailed_w1',
                       'freq_3mo_tobacco_w1',
                       'freq_3mo_alcohol_w1',
                       'freq_3mo_cannabis_w1',
                       'freq_3mo_inhalants_w1',
                       'freq_3mo_hallucinogens_w1',
                       'freq_3mo_stimulants_w1',
                       'ever_3mo_depressants_w1',
                       'num_sex_partner_drugs_w1',
                       'num_nom_sex_w1',
                       'num_nom_soc_w1',
                       'num_sex_partner_w1',
                       'num_oral_partners_w1',
                       'num_anal_partners_w1',
                       'sex_transact_money_w1',
                       'sex_transact_others_w1',
                       'depression_sum_w1',
                       ]
INTER_DEFAULT = -100

def cleaning(data):
    for i, x in enumerate(data):
        for j, elem in enumerate(x):
            if np.isnan(elem) or elem == 'None':
                data[i][j] = INTER_DEFAULT
    return data


class PatientLoader:

    def __init__(self, config, ind_feature_path, sex_adj_path, venue_adj_path, train_mask_path,
                                 graph_feature_path, psk2index_path, is_train):
        self.config = config
        self.feature_path = ind_feature_path
        self.sex_adj_path = sex_adj_path
        self.venue_adj_path = venue_adj_path
        self.mask_path = train_mask_path
        self.graph_feature_path = graph_feature_path
        self.psk2index_path = psk2index_path
        self.labels = []
        self.features = []
        self.graph_features = []
        self.indices = []
        self.masks = []
        self.biases = []
        self.adj = []
        self.psk2index = {}
        self.dataset = []
        self.datasize = 0
        self.feature_size = 0
        self.is_train = is_train

    def load(self):
        self.load_graph_features()
        self.load_attributes()
        if self.is_train:
            self.load_psk2index()
            self.load_adj()
        self.load_mask()
        self.mask_labels()
        # print(self.labels[:50])
        # input()
        self.features = self.features[np.newaxis]
        self.labels = self.labels[np.newaxis]
        self.masks = self.masks[np.newaxis]
        self.dataset = list(zip(self.features, self.labels, self.masks))
        print('feature shape:', self.features.shape)
        self.datasize = self.features.shape[0]
        print("datasize:", self.datasize)

    def load_attributes(self):
        attr_indices = []
        with open(self.feature_path) as ifile:
            ln = 0
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                if ln == 0:
                    header = row
                    attribute2index = {k: i for i, k in enumerate(header)}
                    attr_indices = [attribute2index[a] for a in selected_attributes]
                else:
                    # for hiv label
                    index, label_w1, label_w2, attributes = int(row[0]), int(row[1]), int(row[3]), row
                    # for syphilis label
                    # index, label_w1, label_w2, attributes = int(row[0]), int(row[2]), int(row[4]), row

                    attributes = [attributes[i] for i in attr_indices]
                    attributes = self.recode_attributes(attributes)
                    attributes = [float(a) if a != "" and a != "NA" else -100.0 for a in attributes]
                    graph_attributes = self.graph_features[ln - 1]
                    attributes.extend(graph_attributes)
                    label = self.make_label_from_two_wave(label_w1, label_w2)
                    # print(label, attributes)
                    # input()
                    self.labels.append(label)
                    self.features.append(attributes)
                ln += 1

        # for generating weights for drawing, still some problem !!
        # feature_file_path = './node_features.pkl'
        # pkl.dump(self.features, open(feature_file_path, 'wb'))
        # label_path = './node_labels.pkl'
        # pkl.dump(self.labels, open(label_path, 'wb'))

        #self.load_graph_features()

        # duplicate X and Y for both sex graph and venue graph
        self.labels += self.labels
        self.features += self.features

        self.labels = np.asarray(self.labels)
        self.labels = self.np_to_onehot(self.labels, self.config.num_classes)

        self.features = np.asarray(self.features, dtype=float)
        self.feature_size = self.features.shape[1]

    def load_mask(self):
        print('load mask from %s' % self.mask_path)
        with open(self.mask_path) as ifile:
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                self.masks.append(int(row[0]))
        
        # create mask for venue graph (all 0), only keep sex graph
        # mask is for calculating evaluation metrics, which can be done only on one network
        empty_masks = [0] * len(self.masks)
        self.masks += empty_masks

        self.masks = np.asarray(self.masks)

    def load_graph_features(self):
        print('load graph from %s' % self.graph_feature_path)
        with open(self.graph_feature_path) as ifile:
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                if REDUCE_GRAPH_FEATURES:
                    self.graph_features.append([float(row[i]) for i in range(6)])
                else:
                    self.graph_features.append([float(row[i]) for i in range(8)])

    def load_psk2index(self):
        ln = 0
        with open(self.psk2index_path) as ifile:
            for row in csv.reader(ifile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
                if ln == 0:
                    ln += 1
                else:
                    psk, index = row[0], row[1]
                    self.psk2index[int(psk)] = int(index)

    def load_adj(self):
        '''
        if '.csv' in self.sex_adj_path:
            self.adj_sex = load_adj_csv(self.sex_adj_path, self.psk2index)
        elif '.xls' in self.sex_adj_path:
            self.adj_sex = load_adj_excel(self.sex_adj_path, self.psk2index)
        '''
        self.adj_sex, _ = pkl.load(open(self.sex_adj_path, "rb"))

        self.adj_venue, _, _, _, patient2venue, _, _, _ = pkl.load(open(self.venue_adj_path, "rb"))

        # self.config.venue_thres = 0

        self.adj_venue = self.stratify_venue_matrix(self.adj_venue, self.config.venue_thres)

        num_nodes = self.config.num_nodes

        ## build supra graph
        supra_graph = np.zeros((num_nodes*2, num_nodes*2), dtype=int)
        supra_graph[:num_nodes, :num_nodes] = self.adj_sex
        supra_graph[num_nodes:, num_nodes:] = self.adj_venue

        # subgraph connections
        for i in range(num_nodes, num_nodes*2):
            supra_graph[i, i - num_nodes] = 1
            supra_graph[i - num_nodes, i] = 1

        self.adj_supra = supra_graph[np.newaxis]
        self.biases_supra = adj_to_bias(self.adj_supra, [num_nodes*2], nhood=1)

        print('supra adjacent matrix shape:', self.adj_supra.shape)

    def get_datasize(self):
        return self.datasize

    def get_dataset(self):
        return self.dataset

    def get_feature_size(self):
        return self.feature_size

    def next_batch(self, prev_idx):
        """
        the next batch of data for training
        :param prev_idx:
        :return:
        """
        b = self.config.batch_size
        upper = np.min([self.datasize, b * (prev_idx + 1)], axis=0)
        yield self.dataset[b * prev_idx: upper]

    def mask_labels(self):
        for i, _ in enumerate(self.labels):
            mask = self.masks[i]
            if int(mask) == 0:
                self.labels[i] = np.asarray([0]*self.config.num_classes)

    @staticmethod
    def np_to_onehot(targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])

    @staticmethod
    def make_label_from_two_wave(label_w1, label_w2):
        return label_w1 or label_w2

    @staticmethod
    def recode_attributes(attributes):
        new_attributes = []
        for i, name in enumerate(selected_attributes):
            value = attributes[i]
            if value == 'NA' or value == '':
                new_attributes.append(value)
                continue
            if 'sexual_identity' in name:
                if float(value) == 1 or float(value) == 3:
                    new_attributes.append(1.0)
                else:
                    new_attributes.append(0.0)
            elif name == 'education':
                if float(value) <= 2:
                    new_attributes.append(1.0)
                else:
                    new_attributes.append(0.0)
            elif 'age_jailed' in name:
                if value == '12 or younger':
                    new_attributes.append(12.0)
                else:
                    new_attributes.append(value)
            else:
                new_attributes.append(value)
        return new_attributes

    @staticmethod
    def stratify_venue_matrix(matrix, thres):
        new_matrix = []
        for row in matrix:
            new_matrix.append([1 if i >= thres else 0 for i in row])
        return np.asarray(new_matrix)

