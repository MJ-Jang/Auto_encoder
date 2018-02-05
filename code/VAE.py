####################################################
# RNN-Variational Autoencoder model
#  - Author: Myeongjun Jang
#  - email: xkxpa@korea.ac.kr
#  - git: https://github.com/MJ-Jang
####################################################



################################################## 1. Import modules ###################################################
import gensim
import numpy as np
import code.helper as hp
from code.utils import *
import tensorflow as tf
import os
import time
########################################################################################################################

class VAE:

    def __init__(self,word2vec_path,hidden_dim,latent_dim):

        # word properties
        self.word2vec_model = gensim.models.Word2Vec.load(word2vec_path)
        self.vocab_size = len(self.word2vec_model.wv.index2word) + 1
        self.word_vec_dim = self.word2vec_model.vector_size
        self.lookup = [[0.] *self.word_vec_dim] + [x for x in self.word2vec_model.wv.syn0]

        self.PAD = 0
        self.EOS = self.word2vec_model.wv.vocab['EOS'].index + 1

        # model hyper-parameters
        self.input_embedding_size = self.word_vec_dim
        self.encoder_hidden_units = hidden_dim
        self.latent_units = latent_dim
        self.decoder_hidden_units = self.latent_units

        # placeholders
        # input, target shape : [max_time, batch_size]
        # sequence length shape : [batch_size]
        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32,name='encoder_inputs')
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32,name='decoder_targets')
        self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32,name='decoder_inputs')
        self.encoder_sequence_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='en_sequence_len')
        self.decoder_sequence_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='de_sequence_len')

        self.learning_rate = tf.placeholder(shape=(),dtype=tf.float32,name = "learning_rate")

        # Embedding matrix
        # denote lookup table as placeholder
        self.embeddings = tf.placeholder(shape=(self.vocab_size, self.input_embedding_size), dtype=tf.float32)
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)

    def params(self):
        self.weights = {
            'mu_w': tf.get_variable("mu_w", shape=[self.encoder_hidden_units * 2, self.latent_units],
                                    initializer=tf.contrib.layers.xavier_initializer()),
            'sd_w' : tf.get_variable("sd_w", shape=[self.encoder_hidden_units * 2, self.latent_units],
                                  initializer=tf.contrib.layers.xavier_initializer()),
        }

        self.bias = {
            'mu_b' : tf.Variable(tf.random_normal([self.latent_units], stddev=0.1), name="mu_b"),
            'sd_b' : tf.Variable(tf.random_normal([self.latent_units], stddev=0.1), name="sd_b"),

        }

    # Encoder
    def encoder(self,name):
        # sentence encoder
        with tf.variable_scope("encoder"):
            self.encoder_cell_fw = tf.contrib.rnn.GRUCell(self.encoder_hidden_units)
            self.encoder_cell_bw = tf.contrib.rnn.GRUCell(self.encoder_hidden_units)

            # only need final state encoder
            # semantic vector is final state of encoder
            _, self.encoder_final_state = tf.nn.bidirectional_dynamic_rnn(self.encoder_cell_fw,self.encoder_cell_bw,
                                                                                   inputs=self.encoder_inputs_embedded,
                                                                                   sequence_length=self.encoder_sequence_len,
                                                                                   dtype=tf.float32, time_major=True,scope=name)
            self.final_concat = tf.concat(self.encoder_final_state, axis=1)
        # Mu encoder
        with tf.variable_scope("enc_mu"):
            self.enc_mu = tf.matmul(self.final_concat, self.weights['mu_w']) + self.bias['mu_b']
        # Sigma encoder
        with tf.variable_scope("enc_sd"):
            self.enc_sd = tf.matmul(self.final_concat, self.weights['sd_w']) + self.bias['sd_b']

        # Sample epsilon
        epsilon = tf.random_normal(tf.shape(self.enc_sd), name='epsilon')

        # Sample latent variable
        std_encoder = tf.exp(.5 * self.enc_sd)

        # Compute KL divergence (latent loss)
        self.KLD = -.5 * tf.reduce_sum(1. - tf.square(self.enc_sd) - tf.square(self.enc_mu) + tf.log(tf.square(self.enc_sd) + 1e-8),1)

        # Generate z(latent)
        # z = mu + (sigma * epsilon)
        self.z = self.enc_mu + tf.multiply(std_encoder, epsilon)

    # Decoder
    def decoder(self,name):
        with tf.variable_scope("decoder"):
            self.decoder_cell = tf.contrib.rnn.GRUCell(self.decoder_hidden_units)

            self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(self.decoder_cell,self.decoder_inputs_embedded,
                                                                               sequence_length = self.decoder_sequence_len,
                                                                               initial_state = self.z,dtype=tf.float32,
                                                                               time_major=True, scope=name,)


    def model(self,softmax_sampling_size,softmax_name,bias_name):
        self.decoder_softmax_weight = tf.get_variable(softmax_name, shape=[self.vocab_size, self.decoder_hidden_units],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        self.decoder_softmax_bias = tf.Variable(tf.random_normal([self.vocab_size], stddev=0.1),name=bias_name)

        # sampling softmax cross entropy loss
        # make batch to flat for easy calculation
        self.sampled_softmax_cross_entropy_loss = tf.nn.sampled_softmax_loss(weights=self.decoder_softmax_weight,
                                                                        biases=self.decoder_softmax_bias,
                                                                        labels=tf.reshape(self.decoder_targets, [-1, 1]),
                                                                        inputs=tf.reshape(self.decoder_outputs,
                                                                                          [-1, self.decoder_hidden_units]),
                                                                        num_sampled = softmax_sampling_size,
                                                                        num_classes = self.vocab_size, num_true=1)
        self.BCE = tf.reduce_sum(self.sampled_softmax_cross_entropy_loss)
        self.total_loss = tf.reduce_mean(self.KLD + self.BCE)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

    def log_and_saver(self,log_path,model_path,sess):
        # log
        self.loss_sum = tf.summary.scalar("Loss", self.total_loss)
        self.summary = tf.summary.merge_all()

        self.writer_tr = tf.summary.FileWriter(log_path + "/train", sess.graph)
        self.writer_test = tf.summary.FileWriter(log_path+"/test", sess.graph)

        # saver
        self.dir = os.path.dirname(os.path.realpath(model_path))

    def saver(self):
        self.all_saver = tf.train.Saver()

    # feed_dict function
    def next_feed(self,batch,learning_rate):
        self.encoder_inputs_, self.en_seq_len_ = hp.batch(batch)
        self.decoder_targets_, self.de_seq_len_ = hp.batch([(sequence) + [self.EOS] for sequence in batch])
        self.decoder_inputs_, _ = hp.batch([[self.EOS] + (sequence) for sequence in batch])

        return {
            self.encoder_inputs: self.encoder_inputs_,
            self.decoder_inputs: self.decoder_inputs_,
            self.decoder_targets: self.decoder_targets_,
            self.embeddings: self.lookup,
            self.encoder_sequence_len: self.en_seq_len_,
            self.decoder_sequence_len: self.de_seq_len_,
            self.learning_rate : learning_rate,
        }

    def variable_initialize(self,sess):
        sess.run(tf.global_variables_initializer())

    def train(self,corpus_train,corpus_val,batch_size,n_epoch,init_lr,sess):

        print("Start train !!!!!!!")

        count_t = time.time()

        for i in range(n_epoch):
            for start, end in zip(range(0, len(corpus_train), batch_size),range(batch_size, len(corpus_train), batch_size)):

                batch_time_50 = time.time()

                global_step = i * int(len(corpus_train) / batch_size) + int(start / batch_size + 1)

                ## training
                # regulate learning rate and KLD weight
                if (i+1) < 10:
                    fd = self.next_feed(corpus_train[start:end], learning_rate=init_lr)
                else:
                    fd = self.next_feed(corpus_train[start:end], learning_rate = init_lr*0.1)

                s_tr, _, l_tr = sess.run([self.summary, self.train_op, self.total_loss], feed_dict=fd)
                self.writer_tr.add_summary(s_tr, global_step)

                # validation
                tst_idx = np.arange(len(corpus_val))
                np.random.shuffle(tst_idx)
                tst_idx = tst_idx[0:batch_size]

                if (i+1) < 10:
                    fd_tst = self.next_feed(np.take(corpus_val,tst_idx,0), learning_rate=init_lr)
                else:
                    fd_tst = self.next_feed(np.take(corpus_val,tst_idx,0), learning_rate = init_lr * 0.1)

                s_tst, l_tst = sess.run([self.summary, self.total_loss], feed_dict=fd_tst)
                self.writer_test.add_summary(s_tst, global_step)

                if start == 0 or int(start / batch_size + 1) % 50 == 0:
                    print("Iter", int(start / batch_size + 1), " Training Loss:", l_tr, "Test loss : ", l_tst, "Time : ",time.time() - batch_time_50)

            if (i + 1) % 10 == 0:
                savename = self.dir + "net-" + str(i + 1) + ".ckpt"
                self.all_saver.save(sess=sess, save_path=savename)

            print("epoch : ", i + 1, "loss : ", l_tr, "Test loss : ", l_tst)

        print("Running Time : ", time.time() - count_t)
        print("Training Finished!!!")

    def load_model(self,model_path,model_name,sess):
        restorename = model_path+"/"+model_name
        self.all_saver.restore(sess,restorename)

