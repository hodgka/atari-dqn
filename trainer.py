from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from utils import *

FLAGS = tf.app.flags.FLAGS


class ModelTrainer(object):
    def __init__(self, model):

        self.model_dir = FLAGS.model_dir
        self.use_gpu = FLAGS.use_gpu
        self.device_id = FLAGS.device_id
        self.iterations = FLAGS.iterations

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if self.use_gpu:
            device_str = '/gpu:' + str(self.device_id)
        else:
            device_str = '/cpu:0'

        with tf.device(device_str):
            self.global_step = tf.get_variable("global_step", [],
                                               initializer=tf.constant_initializer(0),
                                               trainable=False)
            self.model = model()
            self.optimizer = self.get_optimizer().minimize(self.model.loss, global_step=self.global_step)

    def train(self):
        init = tf.global_variables_initializer()
        summarize = tf.summary.merge_all()
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)
            summary_writer = tf.summary.FileWriter(self.model_dir, sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            iterations = 1
            try:
                while not coord.should_stop():
                    _, loss = sess.run([self.optimizer, self.model.loss])
                    print("Step {}, loss{:.2f}".format(iterations, loss))

                    if iterations % 10 == 0:
                        summary_str = sess.run(summarize)
                        summary_writer.add_summary(summary_str, iterations)

                    if iterations % 10000 == 0:
                        checkpoint_path = os.path.join(self.model_dir, "model.ckpt")
                        saver.save(sess, checkpoint_path, global_step=iterations)
                    iterations += 1
            except tf.errors.OutOfRangeError:
                print("Done training")
            finally:
                coord.request_stop()
                coord.join(threads)

    def get_optimizer(self, flag, learning_rate, **kwargs):
        '''
        select optimizer to use. defaults to RMS Prop
        '''
        optimizer = tf.train.RMSPropOptimizer(learning_rate, **kwargs)
        if flag.lower() == 'adagrad':
            optimizer = tf.train.AdagraDAOptimizer(learning_rate, **kwargs)
        elif flag.lower() == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate, **kwargs)
        elif flag.lower() == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate, **kwargs)
        return optimizer
