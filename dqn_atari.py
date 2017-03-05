from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
# import numpy as np
import random
import tensorflow as tf


env = gym.make("Breakout-v0")
env.reset()

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("iterations", 20, "Number of iterations to steps.")
tf.app.flags.DEFINE_boolean("use_gpu", True, "Use GPU for training.")
tf.app.flags.DEFINE_integer("device_id", 0, "ID of GPU to use")


ACTIONS = {'left', 'right'}

for i in range(20):
    pass
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())


class Environment(object):
    def __init__(self):
        self.env = gym.make(FLAGS.env_name)
        _w, _h, = FLAGS.screen_width, FLAGS.screen_height
        self.action_repeat, self.start = FLAGS.action_repeat, FLAGS.random_state

        self.display = FLAGS.display
        self.dims = (_w, _h)
        self._screen = None
        self.reward = 0
        self.terminal = True

    def new_game(self, from_random_game=False):
        if self.lives == 0:
            self._screen = self.env.reset()
        self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def new_random_game(self):
        self.new_game(True)
        for _ in range(random.randint(0, self.random_start - 1)):
            self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def _step(self, action):
        self._screen, self.reward, self.terminal, _ = self.env.step(action)

    def _random_step(self):
        action = self.env.action_space.sample()
        self._step(action)

    
