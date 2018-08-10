#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on July 2018

@author: nicola ancona
"""

import tensorflow as tf
import numpy as np
import csv
import os.path
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
MIN_BUFFER_SIZE = 20000
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
# Test flag
TEST = False


def compute_ou_noise(noise):
    # Solve using Euler-Maruyama method
    noise = noise + OU_THETA * (OU_MU - noise) + OU_SIGMA * np.random.randn(ACTION_DIMENSION)
    return noise


def compute_action(actor, s, noise):
    if TEST:
        action = actor.predict(np.reshape(s, (1, STATE_DIMS)))
    else:
        action = actor.predict(np.reshape(s, (1, STATE_DIMS))) + compute_ou_noise(noise)
    action = np.reshape(action, (ACTION_DIMENSION,))
    action = np.clip(action, -1, 1)
    return action


def write_csv_learn(s0, a, s2, t, r, tr):

    file_exists = os.path.isfile('learn.csv')
    a = np.ndarray.tolist(a)
    r = [r]
    t = [t]
    tr = [tr]

    x = s0 + a + s2 + r + t + tr
    with open('learn.csv', 'a') as csvfile1:
        headers = ['trsxp', 'trsyp', 'trsa', 'lha', 'rha', 'lka', 'rka', 'laa', 'raa', 'trsxv', 'trzv', 'trso', 'lho',
                   'rho', 'lko', 'rko', 'lao', 'rao', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'trsxp+1', 'trsyp+1',
                   'trsa+1', 'lha+1', 'rha+1', 'lka+1', 'rka+1', 'laa+1', 'raa+1', 'trsxv+1', 'trzv+1', 'trso+1',
                   'lho+1', 'rho+1', 'lko+1', 'rko+1', 'lao+1', 'rao+1', 'reward', 'terminal', 'tot_reward']
        wr = csv.writer(csvfile1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            wr.writerow(headers)  # file doesn't exist yet, write a header

        wr.writerow(x)


def write_csv_test(s0, a, s2, t, r, tr):
    file_exists = os.path.isfile('test.csv')
    a = np.ndarray.tolist(a)
    r = [r]
    t = [t]
    tr = [tr]

    x = s0 + a + s2 + r + t + tr
    with open('test.csv', 'a') as csvfile2:
        headers = ['trsxp', 'trsyp', 'trsa', 'lha', 'rha', 'lka', 'rka', 'laa', 'raa', 'trsxv', 'trzv', 'trso', 'lho',
                   'rho', 'lko', 'rko', 'lao', 'rao', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'trsxp+1', 'trsyp+1',
                   'trsa+1', 'lha+1', 'rha+1', 'lka+1', 'rka+1', 'laa+1', 'raa+1', 'trsxv+1', 'trzv+1', 'trso+1',
                   'lho+1', 'rho+1', 'lko+1', 'rko+1', 'lao+1', 'rao+1', 'reward', 'terminal', 'tot_reward']
        wr = csv.writer(csvfile2, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        if not file_exists:
            wr.writerow(headers)  # file doesn't exist yet, write a header

        wr.writerow(x)


def train(sess, actor, critic):
    t = 0  # test counter

    sess.run(tf.global_variables_initializer())

    # initialize actor, critic and replay buffer
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

        if not i % 10 and i > 0 and replay_buffer.size() > MIN_BUFFER_SIZE:
            TEST = True
            t += 1
        else:
            TEST = False
        # initialize noise process
        noise = np.zeros(ACTION_DIMENSION)
        total_episode_reward = 0

        for j in range(MAX_EPISODE_LENGTH):
            s0 = s.current_state()
            a = compute_action(actor, s0, noise)
            # computing next step, reward and terminal
            s2 = s.next_states(s0, a)
            r = s.calc_reward(s2, s0)
            print s.current_state()
            terminal = s.calc_terminal()

            if not TEST:
                replay_buffer.add(np.reshape(s0, (actor.s_dim,)), np.reshape(a, actor.a_dim), r,
                                  terminal, np.reshape(s2, (actor.s_dim,)))

            total_episode_reward += r

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if not TEST:
                if replay_buffer.size() > MIN_BUFFER_SIZE:
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

            if not TEST:
                write_csv_learn(s0, a, s2, terminal, r, total_episode_reward)

            else:
                write_csv_test(s0, a, s2, terminal, r, total_episode_reward)

            if not terminal == 0:
                print t, i, j, total_episode_reward  # printing n of test, n of train, length of the episode,
                                                     # tot ep reward
                break

        s = s.reset()


def main():

    # Initialize the actor, critic
    with tf.Session() as sess:

        actor = Actor(sess, STATE_DIMS, ACTION_DIMENSION, 1, ACTOR_LEARNING_RATE, TAU)
        critic = Critic(sess, STATE_DIMS, ACTION_DIMENSION, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())
        train(sess, actor, critic)


if __name__ == "__main__":
    main()
