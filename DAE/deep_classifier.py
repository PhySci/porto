import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import roc_auc_score
import logging


class DeepClassifier:

    def __init__(self, dae_lr=0.001, nn_lr=0.01, dae_decay=0.99, nn_decay=0.99, L2 = 0.005, batch_size=512,
                 features = 345, dae_hidden = [1500, 1500, 1500], clf_hidden = [1000, 1000],
                 restart=False, verbose=True, keep_prob=1, name = './DAE_model'):
        """
        Create an instance
        :param dae_lr:
        :param nn_lr:
        :param features:
        :param dae_hidden:
        :param clf_hidden:
        :param decay:
        :param momentum:
        :param restart:
        :param verbose:
        :param dropout:
        """

        # Training Parameters
        self.dae_learning_rate = dae_lr
        self.nn_learning_rate = nn_lr
        self.batch_size = batch_size
        self.dae_decay = dae_decay
        self.nn_decay = nn_decay
        self.momentum = 0
        self.L2 = L2
        self.verbose = True
        self.keep_prob = keep_prob
        self.dae_epoch = 100
        self.clf_epoch = 100


        self.pth = name

        self.dae_size= [features]+ dae_hidden + [features]
        self.nn_size = [np.array(self.dae_size[1:-1]).sum()] + clf_hidden + [1]

        tf.reset_default_graph()
        self.inp_features = tf.placeholder(tf.float32, [None, self.dae_size[0]], name='inp_features')
        self.ref_features = tf.placeholder(tf.float32, [None, self.dae_size[0]], name='ref_features')
        self.labels = tf.placeholder(tf.int8, [None, 1], name='labels')
        self.keep_prob_ph = tf.placeholder(tf.float32, [], name='keep_prob_ph')

        self.graph = self.build_graph(self.inp_features)

        self.dae_loss = self.get_dae_loss(self.ref_features)
        self.dae_optimizer = self.get_dae_optimizer()

        self.clf_loss = self.get_clf_loss(self.labels)
        self.clf_optimizer = self.get_clf_optimizer()

        self.saver = tf.train.Saver()
        logging.basicConfig(filename='DAE.log', level=logging.DEBUG, format='%(message)s')
        self.logger = logging.getLogger('DAE_classifier')

        if restart:
            init_op = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init_op)
                self.saver.save(sess, self.pth)
                self.logger.debug('Init and save the model')

    def get_auc(self):
       pass

    def get_dae_loss(self, ref_x):
        #for n in tf.get_default_graph().as_graph_def().node:
        #   print(n.name)
        with tf.name_scope('DAE'):
            x_dae = tf.get_default_graph().get_tensor_by_name("DAE/layer3/output:0")
            loss = tf.reduce_mean(tf.pow(ref_x - x_dae, 2, name='pow2'), name='loss')
        return loss

    def get_dae_optimizer(self):
        with tf.name_scope('dae_optimizer'):
            return tf.train.RMSPropOptimizer(self.dae_learning_rate, decay=self.dae_decay, momentum=self.momentum,
                                             name='dae_optimizer_op'). \
                minimize(self.dae_loss, var_list=tf.get_collection('DAE'))

    def train_dae(self, X):
        noise_list = np.linspace(0.001, 0.1, self.dae_epoch)

        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            self.saver.restore(sess, self.pth)
            train_writer = tf.summary.FileWriter('./train', sess.graph)
            for epoch in range(self.dae_epoch): #enumerate(noise_list):
                noise = 0.1
                l = 0
                for i in range(np.ceil(X.shape[0] / self.batch_size).astype(int)):
                    batch, noise_batch = self.get_batch(X, noise_level=noise)
                    summary, _, t = sess.run([merged, self.dae_optimizer, self.dae_loss],
                                    feed_dict={self.ref_features:batch,
                                               self.inp_features: noise_batch,
                                               self.keep_prob_ph: self.keep_prob})
                    l += t
                msg = '{epoch: 3d}: loss is {l: 2.3f}, noise level is {noise: 2.4f}'.format(epoch=epoch, l=l, noise=noise)
                self.show_msg(msg)

                train_writer.add_summary(summary, epoch)

                if epoch % 10 ==0:
                    self.logger.debug(self.saver.save(sess, self.pth))



            print(self.saver.save(sess, self.pth))

    def get_clf_loss(self, y):
        with tf.name_scope('clf'):
            y_pr = tf.get_default_graph().get_tensor_by_name("NN/layer2/output:0")
            loss_sample =  tf.reduce_mean(tf.losses.log_loss(y, y_pr), name='clf_loss')
            loss_reg = tf.nn.l2_loss(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            return loss_sample+self.L2*loss_reg

    def get_clf_optimizer(self):
        with tf.name_scope('clf_optimizer'):
            return tf.train.RMSPropOptimizer(self.nn_learning_rate, decay=self.nn_decay, momentum=self.momentum,
                                             name='clf_optimizer_op'). \
                minimize(self.clf_loss, var_list=tf.get_collection('clf'))

    def train_clf(self, X, y, X_val=None, y_val=None, restart = False, L2 = 0.005):

        self.logger.debug('Classifier training')
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('./train', sess.graph)
            if restart:
                tf.initialize_variables(tf.get_collection('clf'))
            self.saver.restore(sess, self.pth)
            for epoch in range(self.clf_epoch):
                l = 0
                for i in range(np.ceil(X.shape[0] / self.batch_size).astype(int)):
                    batch_x, batch_y = get_train_batch(X, y, batch_size=self.batch_size)
                    summary, _, t = sess.run([merged, self.clf_optimizer, self.clf_loss],
                                    feed_dict={self.inp_features: batch_x,
                                               self.labels:batch_y,
                                               self.keep_prob_ph: self.keep_prob})
                    l += t
                msg = '{epoch: 3d}: {l: 2.3f}'.format(epoch=epoch, l=l)
                self.show_msg(msg)
                if (epoch % 2 ==0) or epoch==(self.clf_epoch-1):
                    y_pr = sess.run(self.graph, feed_dict={self.inp_features: X_val, self.keep_prob_ph: 1})
                    msg = 'ROC-AUC is {v: 2.3f}'.format(v=roc_auc_score(y_val, y_pr))
                    self.show_msg(msg)
                    self.logger.debug(self.saver.save(sess, self.pth))

                train_writer.add_summary(summary, epoch)

            print(self.saver.save(sess, self.pth))

    def build_graph(self, inp):
        # Define DAE
        with tf.name_scope('DAE'):
            with tf.name_scope('layer0'):
                dae_w0 = tf.get_variable(shape=[self.dae_size[0], self.dae_size[1]], name='dae_w0')
                dae_b0 = tf.get_variable(shape=[self.dae_size[1]], name='dae_b0')
                dae_layer0 = tf.nn.sigmoid(tf.add(tf.matmul(inp, dae_w0), dae_b0), name='output')
                tf.add_to_collection("DAE", dae_w0)
                tf.add_to_collection("DAE", dae_b0)

                tf.summary.histogram('dae_w0', dae_w0)
                tf.summary.histogram('dae_b0', dae_b0)
                tf.summary.histogram('dae_layer0', dae_layer0)

            with tf.name_scope('layer1'):
                dae_w1 = tf.get_variable(shape=[self.dae_size[1], self.dae_size[2]], name='dae_w1')
                dae_b1 = tf.get_variable(shape=[self.dae_size[2]], name='dae_b1')
                dae_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(dae_layer0, dae_w1), dae_b1), name='output')
                tf.add_to_collection("DAE", dae_w1)
                tf.add_to_collection("DAE", dae_b1)

            with tf.name_scope('layer2'):
                dae_w2 = tf.get_variable(shape=[self.dae_size[2], self.dae_size[3]], name='dae_w2')
                dae_b2 = tf.get_variable(shape=[self.dae_size[3]], name='dae_b2')
                dae_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(dae_layer1, dae_w2), dae_b2), name='output')
                tf.add_to_collection("DAE", dae_w2)
                tf.add_to_collection("DAE", dae_b2)

            with tf.name_scope('layer3'):
                dae_w3 = tf.get_variable(shape=[self.dae_size[3], self.dae_size[4]], name='dae_w3')
                dae_b3 = tf.get_variable(shape=[self.dae_size[4]], name='dae_b3')
                dae_layer3 = tf.add(tf.matmul(dae_layer2, dae_w3), dae_b3, name='output')
                tf.add_to_collection("DAE", dae_w3)
                tf.add_to_collection("DAE", dae_b3)

        # Define Neural Network Classifier
        with tf.name_scope('NN'):
            nn_inp = tf.concat([dae_layer0, dae_layer1, dae_layer2], axis=1, name='concat')
            regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)
            with tf.name_scope('layer0'):
                nn_b0 = tf.get_variable(shape=[self.nn_size[1]], name='clf_b0')
                nn_w0 = tf.get_variable(shape=[self.nn_size[0], self.nn_size[1]], name='clf_w0', regularizer=regularizer)
                a0 = tf.nn.relu(tf.add(tf.matmul(nn_inp, nn_w0), nn_b0), name='output')
                nn_layer0 = tf.nn.dropout(a0, self.keep_prob_ph, name='dropout_0')
                tf.add_to_collection("clf", nn_w0)
                tf.add_to_collection("clf", nn_b0)
                tf.summary.histogram('nn_w0', nn_w0)
                tf.summary.histogram('nn_layer0', nn_layer0)
            with tf.name_scope('layer1'):
                nn_b1 = tf.get_variable(shape=[self.nn_size[2]], name='clf_b1')
                nn_w1 = tf.get_variable(shape=[self.nn_size[1], self.nn_size[2]], name='clf_w1', regularizer=regularizer)
                a1 = tf.nn.relu(tf.add(tf.matmul(nn_layer0, nn_w1), nn_b1), name='output')
                nn_layer1 = tf.nn.dropout(a1, keep_prob=self.keep_prob_ph, name='dropout_1')
                tf.add_to_collection("clf", nn_w1)
                tf.add_to_collection("clf", nn_b1)
            with tf.name_scope('layer2'):
                nn_b2 = tf.get_variable(shape=[self.nn_size[3]], name='clf_b2')
                nn_w2 = tf.get_variable(shape=[self.nn_size[2], self.nn_size[3]], name='clf_w2', regularizer=regularizer)
                a2 = tf.nn.sigmoid(tf.add(tf.matmul(nn_layer1, nn_w2), nn_b2), name='output')
                y_pr = tf.nn.dropout(a2, keep_prob=self.keep_prob_ph, name='dropout_2')
                tf.add_to_collection("clf", nn_w2)
                tf.add_to_collection("clf", nn_b2)
        return y_pr

    def get_batch(self, features, noise_level = 0.1):
        n_features = features.shape[1]
        idx = np.random.randint(0, features.shape[0], self.batch_size)
        new_set = features.iloc[idx].values
        n_perm = int(noise_level*self.batch_size*n_features)
        rows = list(np.random.randint(0, self.batch_size, n_perm))
        cols = list(np.random.randint(0, n_features, n_perm))
        new_rows = list(np.random.randint(0, features.shape[0], n_perm))
        noise_set = new_set.copy()
        noise_set[rows, cols] = features.values[new_rows, cols]
        return new_set, noise_set

    def predict(self, X):
        with tf.Session() as sess:
            self.saver.restore(sess, self.pth)
            y_pr = sess.run(self.graph, feed_dict={self.inp_features: X, self.keep_prob_ph: 1})
        return y_pr

    def show_msg(self, msg):
        self.logger.debug(msg)
        if self.verbose:
            print(msg)

def get_train_batch(X, y, batch_size = 100):
    idx = np.random.randint(0, X.shape[0], batch_size)
    return X.iloc[idx].values, y.iloc[idx].values[:, np.newaxis]


 #graph = tf.get_default_graph()
        #x_noise = graph.get_tensor_by_name("x_noise:0")
        #x_out_op = graph.get_tensor_by_name("DAE/layer4/x_out_op:0")


#def get_optimizer(self):
    #    with tf.name_scope('optimizer'):
    #        global_step = tf.Variable(0, trainable=False)
    #        lr = tf.train.exponential_decay(self.learning_rate, global_step, 1, 0.995)
    #        #, global_step=global_step
    #        return tf.train.GradientDescentOptimizer(self.learning_rate, name='optimizer_op').\
    #            minimize(self.loss)


    """ 
    def train(self, features, target):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(self.num_epoch):
                l = 0
                for i in range(np.ceil(features.shape[0] / self.batch_size).astype(int)):
                    batch_x, batch_y = get_train_batch(features, target, self.batch_size)
                    sess.run(self.optimizer, feed_dict={self.features: batch_x, self.labels: batch_y})

                if epoch % 1 == 0:
                    l = sess.run(self.loss, feed_dict={self.features: batch_x, self.labels: batch_y})
                    print('{epoch: 2d}: {loss: 3.2f}'.format(epoch=epoch, loss = l))
    """

# print(tf.get_collection('DAE'))
# print(tf.get_collection('clf'))
# return y_pr




