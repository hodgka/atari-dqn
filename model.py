import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


class QLearner(object):
    def __init__(self, env):
        with tf.variable_scope('qlearner'):
            self.X = env
            self.build_model(env)

    def loss(self,):
        pass

    def build_model(env):
        pass
