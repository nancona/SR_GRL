#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on July 2018

@author: nicola ancona
"""

import tensorflow as tf
import numpy as np
import tflearn
import math
from ReplayBuffer import ReplayBuffer
from computingThings import Step
from models import Models

# ==========================
#   Training Parameters
# ==========================
# Max training steps
# MAX_EPISODES = 50000
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
# ???????
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

def ou_noise(theta, mu, sigma, noise, dims):
    # Solve using Euler-Maruyama method
    noise = noise + theta * (mu - noise) + sigma * np.random.randn(dims)
    return noise

def compute_action(actor, mod_state, noise):
    action = actor.predict(np.reshape(mod_state, (1, actor.s_dim))) + ou_noise(OU_THETA, OU_MU, OU_SIGMA, noise, ACTION_DIMENSION)
    action = np.reshape(action, (ACTION_DIMENSION,))
    action = np.clip(action, -1, 1)
    return action

def train(args, actor, critic, actor_noise):

    actor.update_target_network()
    critic.update_target_network()
    replay_buffer = ReplayBuffer(int(args['buffer size']), int(args['random_seed']))

    for i in range(int(args['max_episodes'])):

        s = Models.reset()
        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            a = compute_action(actor, s, noise)
            s2, r, terminal, info = env.step(a[0])
            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, actor.a_dim), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = ReplayBuffer.sample_batch(int(args['minibatch_size']))

                # calculate targets
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args(['minibatch_size']))):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r