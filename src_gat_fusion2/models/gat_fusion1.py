"""
GAT_fusion: combine the logits of two networks together to make predictions
"""

import numpy as np
import tensorflow as tf

from src_gat_fusion2.models import layers
from src_gat_fusion2.base.base_model import BaseModel
from src_gat_fusion2.base.checkmate import BestCheckpointSaver
from src_gat_fusion2.models.metrics import *

tf.random.set_random_seed(1234)


class GAT(BaseModel):
    def __init__(self, config):
        super(GAT, self).__init__(config)
        with tf.name_scope('input'):
            self.features = tf.placeholder(dtype=tf.float32,
                                    shape=(self.config.batch_size, self.config.num_nodes, self.config.feature_size),
                                           name='features_ph')  # input feature
            self.bias_in_sex = tf.placeholder(dtype=tf.float32,
                                     shape=(self.config.batch_size, self.config.num_nodes, self.config.num_nodes),
                                          name='adj_bias_sex')  # bias vector from sex adj
            self.bias_in_venue = tf.placeholder(dtype=tf.float32,
                                     shape=(self.config.batch_size, self.config.num_nodes, self.config.num_nodes),
                                          name='adj_bias_venue')  # bias vector from venue adj

            self.labels = tf.placeholder(dtype=tf.int32,
                                    shape=(self.config.batch_size, self.config.num_nodes, self.config.num_classes),
                                         name='labels_ph')  # labels
            self.masks_in = tf.placeholder(dtype=tf.int32, shape=(self.config.batch_size, self.config.num_nodes),
                                         name='masks_ph')  # consider the info or not
            
            self.attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')  # attention dropout
            self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')  # feed forward dropout
            self.is_training = tf.placeholder(dtype=tf.bool, shape=())

            self.initializer = tf.random_normal_initializer(stddev=0.1)

        self.build_model()
        self.init_saver()

    def build_model(self):
        if self.config.nonlinearity == 'relu':
            activation = tf.nn.relu

        logits, att_weights_sex, att_weights_venue, W_sex, W_venue = \
            self.inference_twograph_middle_fusion(self.features, self.config.num_classes,
                                                    self.attn_drop, self.ffd_drop,
                                                    bias_mat_sex=self.bias_in_sex, bias_mat_venue=self.bias_in_venue,
                                                    hid_units=self.config.hid_units, n_heads=self.config.num_heads,
                                                    residual=self.config.residual, activation=activation)

        with tf.name_scope("loss"):
            logits_reshape = tf.reshape(logits, [-1, self.config.num_classes])
            labels_reshape = tf.reshape(self.labels, [-1, self.config.num_classes])
            masks_reshape = tf.reshape(self.masks_in, [-1])
            self.probs = tf.sigmoid(logits_reshape)
            self.preds = tf.cast(tf.argmax(logits_reshape, 1), tf.int32)
            self.loss = masked_softmax_cross_entropy(logits_reshape, labels_reshape, masks_reshape)
            self.accuracy = masked_accuracy(logits_reshape, labels_reshape, masks_reshape)
            self.f1 = masked_micro_f1(logits_reshape, labels_reshape, masks_reshape)

            # weights to generate neighbor contributions
            self.att_weights_sex = att_weights_sex
            self.att_weights_venue = att_weights_venue
            self.W_sex = W_sex
            self.W_venue = W_venue

        # def training(self, loss, lr, l2_coef):
        with tf.name_scope("train_op"):
            # weight decay
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                               in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.config.l2_coef

            # optimizer
            opt = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)

            # training op
            self.train_step = opt.minimize(self.loss + lossL2)

    def inference(self, inputs, nb_classes, attn_drop, ffd_drop,
                  bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        # the hidden vector for each node is the concatenation of multiple attention head
        attns = []
        coefs = []
        for _ in range(n_heads[0]):
            head, coef = layers.attn_head(inputs, bias_mat=bias_mat,
                                          out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            # head: [1, num_nodes, hid_unit], coef: [1, num_nodes, num_nodes]
            attns.append(head)
            coefs.append(coef)


        h_1 = tf.concat(attns, axis=-1) # h_1: [1, num_nodes, hid_unit*num_heads[0]]
        att_weights = tf.concat(coefs, axis=-1) # att_weights: [1, num_nodes, num_nodes*num_heads[0]]
        att_weights = tf.reduce_mean(coefs, -1) # att_weights: [1, num_nodes, num_nodes]
        
        #print(att_weights.shape)
        #input()

        # if more transformer layers added, add residual to them
        for i in range(1, len(hid_units)):
            print("att layer {}".format(i))
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                head, _ = layers.attn_head(h_1, bias_mat=bias_mat,
                                              out_sz=hid_units[i], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=residual)
                attns.append(head)
            h_1 = tf.concat(attns, axis=-1)


        # the output head number is 1
        out = []
        for i in range(n_heads[-1]):
            head, _ = layers.attn_head(h_1, bias_mat=bias_mat,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            out.append(head)
        # ? how to get attention weights and feature weights?
        logits = tf.add_n(out) / n_heads[-1]
        return logits



    def inference_twograph_early_fusion(self, inputs, nb_classes, attn_drop, ffd_drop,
                  bias_mat_sex, bias_mat_venue, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        """
        fusion at the 2nd layer
        :param inputs:
        :param nb_classes:
        :param attn_drop:
        :param ffd_drop:
        :param bias_mat_sex:
        :param bias_mat_venue:
        :param hid_units:
        :param n_heads:
        :param activation:
        :param residual:
        :return:
        """
        # the hidden vector for each node is the concatenation of multiple attention head
        # the first attention layer for sex network
        attns = []
        coefs = []
        for _ in range(n_heads[0]):
            head, coef = layers.attn_head(inputs, bias_mat=bias_mat_sex,
                                          out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            # head: [1, num_nodes, hid_unit], coef: [1, num_nodes, num_nodes]
            attns.append(head)
            coefs.append(coef)

        h_1_sex = tf.concat(attns, axis=-1) # h_1: [1, num_nodes, hid_unit*num_heads[0]]

        # ---
        # the first attention layer for venue network
        attns = []
        coefs = []
        for _ in range(n_heads[0]):
            head, coef = layers.attn_head(inputs, bias_mat=bias_mat_venue,
                                          out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            # head: [1, num_nodes, hid_unit], coef: [1, num_nodes, num_nodes]
            attns.append(head)
            coefs.append(coef)

        h_1_venue = tf.concat(attns, axis=-1)  # h_1: [1, num_nodes, hid_unit*num_heads[0]]

        # --- fusion ---

        h_1 = tf.concat([h_1_sex, h_1_venue], axis=-1)

        # if more transformer layers added, add residual to them
        for i in range(1, len(hid_units)):
            print("att layer {}".format(i))
            attns = []
            for _ in range(n_heads[i]):
                head, _ = layers.attn_head(h_1, bias_mat=bias_mat_sex,
                                           out_sz=hid_units[i], activation=activation,
                                           in_drop=ffd_drop, coef_drop=attn_drop, residual=residual)
                attns.append(head)
            h_1 = tf.concat(attns, axis=-1)

        # ---

        # the output head number is 1
        out = []
        for i in range(n_heads[-1]):
            head, _ = layers.attn_head(h_1, bias_mat=bias_mat_sex,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            out.append(head)

        logits = tf.add_n(out) / n_heads[-1]
        return logits


    def inference_twograph_middle_fusion(self, inputs, nb_classes, attn_drop, ffd_drop,
                  bias_mat_sex, bias_mat_venue, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        """
        fusion at the 2nd last layer
        :param inputs:
        :param nb_classes:
        :param attn_drop:
        :param ffd_drop:
        :param bias_mat_sex:
        :param bias_mat_venue:
        :param hid_units:
        :param n_heads:
        :param activation:
        :param residual:
        :return:
        """
        # the hidden vector for each node is the concatenation of multiple attention head
        attns = []
        coefs = []
        for _ in range(n_heads[0]):
            head, coef = layers.attn_head(inputs, bias_mat=bias_mat_sex,
                                          out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            # head: [1, num_nodes, hid_unit], coef: [1, num_nodes, num_nodes]
            attns.append(head)
            coefs.append(coef)

        # if more transformer layers added, add residual to them
        for i in range(1, len(hid_units)):
            print("att layer {}".format(i))
            attns = []
            for _ in range(n_heads[i]):
                head, _ = layers.attn_head(h_1_sex, bias_mat=bias_mat_sex,
                                           out_sz=hid_units[i], activation=activation,
                                           in_drop=ffd_drop, coef_drop=attn_drop, residual=residual)
                attns.append(head)
            h_1 = tf.concat(attns, axis=-1)
      
        h_1_sex = tf.concat(attns, axis=-1) # h_1: [1, num_nodes, hid_unit*num_heads[0]]
        att_weights_sex = tf.add_n(coefs)
  
        # --- venue ---

        attns = []
        coefs = []
        for _ in range(n_heads[0]):
            head, coef = layers.attn_head(inputs, bias_mat=bias_mat_venue,
                                          out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            # head: [1, num_nodes, hid_unit], coef: [1, num_nodes, num_nodes]
            attns.append(head)
            coefs.append(coef)
       
        # if more transformer layers added, add residual to them
        for i in range(1, len(hid_units)):
            print("att layer {}".format(i))
            attns = []
            for _ in range(n_heads[i]):
                head, _ = layers.attn_head(h_1_venue, bias_mat=bias_mat_venue,
                                           out_sz=hid_units[i], activation=activation,
                                           in_drop=ffd_drop, coef_drop=attn_drop, residual=residual)
                attns.append(head)
            h_1_venue = tf.concat(attns, axis=-1)
 
        h_1_venue = tf.concat(attns, axis=-1)  # h_1: [1, num_nodes, hid_unit*num_heads[0]]
        att_weights_venue = tf.add_n(coefs)

        # --- fusion ---
        #
        #'''
        W_sex = tf.Variable(tf.zeros([1, self.config.num_nodes, hid_units[-1]]), name='W_sex')
        W_venue = tf.Variable(tf.zeros([1, self.config.num_nodes, hid_units[-1]]), name='W_venue')
        h_1_sex = W_sex*h_1_sex
        h_1_venue = W_venue*h_1_venue
        #'''
        h_1 = tf.concat([h_1_sex, h_1_venue], axis=-1)


        # the output head number is 1
        out = []
        for i in range(n_heads[-1]):
            head, _ = layers.attn_head(h_1, bias_mat=bias_mat_sex,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            out.append(head)

        logits = tf.add_n(out) / n_heads[-1]


        return logits, att_weights_sex, att_weights_venue, W_sex, W_venue


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # save the 5 best ckpts

        best_ckpt_saver = BestCheckpointSaver(
            save_dir=self.config.best_model_dir,
            num_to_keep=1,
            maximize=True
        )
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        self.saver = best_ckpt_saver

    @staticmethod
    def preshape(logits, labels, nb_classes):
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    @staticmethod
    def confmat(logits, labels):
        preds = tf.argmax(logits, axis=1)
        return tf.confusion_matrix(labels, preds)


