#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on July 2018

@author: nicola ancona
"""

import tensorflow as tf
import numpy as np
from ReplayBuffer import ReplayBuffer
from models import Models
from actorNetwork import Actor
from criticNetwork import Critic

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EPISODE_LENGTH = 1010
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
# ===========================
#   Utility Parameters
# ===========================
RANDOM_SEED = 1234
# Size of replay buffer
TRAINING_SIZE = 2000
BUFFER_SIZE = 300000
MINIBATCH_SIZE = 64
MIN_BUFFER_SIZE = 200
# Environment Parameters
ACTION_DIMENSION = 6
ACTION_DIMENSION_GRL = 9
STATE_DIMS = 18
ACTION_BOUND = 1
ACTION_BOUND_REAL = 8.6
# Noise Parameters
NOISE_MEAN = 0
NOISE_VAR = 1
# Ornstein-Uhlenbeck variables
OU_THETA = 0.15
OU_MU = 0
OU_SIGMA = 0.2


def compute_ou_noise(noise):
    # Solve using Euler-Maruyama method
    noise = noise + OU_THETA * (OU_MU - noise) + OU_SIGMA * np.random.randn(ACTION_DIMENSION)
    return noise


def compute_action(actor, s, noise):
    action = actor.predict(np.reshape(s, (1, STATE_DIMS))) + compute_ou_noise(noise)
    action = np.reshape(action, (ACTION_DIMENSION,))
    action = np.clip(action, -1, 1)
    return action


def train(sess, actor, critic):
    sess.run(tf.global_variables_initializer())

    #initialize actor, critic and replay buffer
    actor.update_target_network()
    critic.update_target_network()
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    s = Models(0, 0, -0.101485,
               0.100951, 0.819996, -0.00146549,
               -1.27, 4.11e-6, 2.26e-7,
               0, 0, 0,
               0, 0, 0,
               0, 0, 0)

    print s.current_state()

    for i in range(MAX_EPISODES):

        # initialize noise process
        noise = np.zeros(ACTION_DIMENSION)
        total_episode_reward = 0

        for j in range(MAX_EPISODE_LENGTH):
            s0 = s.current_state()
            a = compute_action(actor, s0, noise)

            #computing next step, reward and terminal
            s2 = s.next_states(s0, a)
            r = s.calc_reward(s2, s0)
            print s.current_state()
            terminal = s.calc_terminal()

            replay_buffer.add(np.reshape(s0, (actor.s_dim,)), np.reshape(a, actor.a_dim), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            total_episode_reward += r

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MIN_BUFFER_SIZE:
                print "reached min buffer size"
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                # calculate targets
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                # ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            if not terminal == 0:
                print i, j, total_episode_reward
                break

        s = s.reset()


def main():

    # Initialize the actor, critic and difference networks

    with tf.Session() as sess:

        actor = Actor(sess, STATE_DIMS, ACTION_DIMENSION, 1,
                             ACTOR_LEARNING_RATE, TAU)
        critic = Critic(sess, STATE_DIMS, ACTION_DIMENSION, CRITIC_LEARNING_RATE, TAU,
                               actor.get_num_trainable_vars())
        train(sess, actor, critic)


if __name__ == "__main__":
    main()
