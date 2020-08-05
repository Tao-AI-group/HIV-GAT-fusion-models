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
                                    shape=(self.config.batch_size, self.config.num_nodes*2, self.config.feature_size),
                                           name='features_ph')  # input feature
            self.bias_in = tf.placeholder(dtype=tf.float32,
                                     shape=(self.config.batch_size, self.config.num_nodes*2, self.config.num_nodes*2),
                                          name='adj_bias')  # bias vector from sex adj
            self.labels = tf.placeholder(dtype=tf.int32,
                                    shape=(self.config.batch_size, self.config.num_nodes*2, self.config.num_classes),
                                         name='labels_ph')  # labels
            self.masks_in = tf.placeholder(dtype=tf.int32, shape=(self.config.batch_size, self.config.num_nodes*2),
                                         name='masks_ph')  # consider the info or not
            self.attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')  # attention dropout
            self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')  # feed forward dropout
            self.is_training = tf.placeholder(dtype=tf.bool, shape=())

            # self.initializer = tf.random_normal_initializer(stddev=0.1)

        self.build_model()
        self.init_saver()

    def build_model(self):
        if self.config.nonlinearity == 'relu':
            activation = tf.nn.relu
        
        logits, att_weights = self.inference(self.features, self.config.num_classes,
                                self.attn_drop, self.ffd_drop,
                                bias_mat=self.bias_in,
                                hid_units=self.config.hid_units, n_heads=self.config.num_heads,
                                residual=self.config.residual, activation=activation)

        with tf.name_scope("loss"):
            logits_reshape = tf.reshape(logits, [-1, self.config.num_classes])
            labels_reshape = tf.reshape(self.labels, [-1, self.config.num_classes])
            masks_reshape = tf.reshape(self.masks_in, [-1])
            self.probs = tf.sigmoid(logits_reshape)
            self.preds = tf.cast(tf.argmax(logits_reshape, 1), tf.int32)
            correct_prediction = tf.equal(tf.argmax(logits_reshape, 1), tf.argmax(labels_reshape, 1))
            self.loss = masked_softmax_cross_entropy(logits_reshape, labels_reshape, masks_reshape)
            self.accuracy = masked_accuracy(logits_reshape, labels_reshape, masks_reshape)
            self.f1 = masked_micro_f1(logits_reshape, labels_reshape, masks_reshape)
       
        with tf.name_scope("weights"):
            att_weights = tf.reshape(att_weights, [-1, self.config.num_nodes*2])
            #att_weights = tf.nn.softmax(att_weights, axis=-1)
            self.att_weights = att_weights

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
            # head: [1, num_nodes, hid], coef: [1, num_nodes, num_nodes]
            attns.append(head)
            coefs.append(coef)

        h_1 = tf.concat(attns, axis=-1) # h_1: [1, num_nodes, hid*num_heads[0]]

        # if more transformer layers added, add residual to them
        # if only one hidden layer, this section will be ignored
        for i in range(1, len(hid_units)):
            print("att layer {}".format(i))
            h_old = h_1
            attns = []
            coefs = []
            for _ in range(n_heads[i]):
                head, coef = layers.attn_head(h_1, bias_mat=bias_mat,
                                              out_sz=hid_units[i], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=residual)
                attns.append(head)
                coefs.append(coef)
            h_1 = tf.concat(attns, axis=-1)
   
        # the output head number is 1
        out = []
        for i in range(n_heads[-1]):
            head, coef = layers.attn_head(h_1, bias_mat=bias_mat,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False)
            out.append(head)
 

        ## the attention weights should be extracted from the connection between inputs and the first attention layer
        # coefs: num_heads*[num_nodes, num_nodes]

        # the attention weights are from the first attention layer 
        att_weights = tf.add_n(coefs) # att_weights: [1, num_nodes, num_nodes]
        att_weights = tf.math.divide(att_weights, n_heads[0]) # att_weights: [1, num_nodes, num_nodes]

        logits = tf.add_n(out) / n_heads[-1]
        
        return logits, att_weights


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


