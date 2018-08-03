#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on July 2018

@author: nancona
"""

import numpy as np
import math as m
import scipy.io as sio

class Models(object):

    def __init__(self, torso_x_position, torso_z_position, torso_alpha, left_hip_alpha, right_hip_alpha,
                 left_knee_alpha, right_knee_alpha, left_ankle_alpha, right_ankle_alpha, torso_x_vel, torso_z_vel,
                 torso_omega, left_hip_omega, right_hip_omega, left_knee_omega, right_knee_omega, left_ankle_omega,
                 right_ankle_omega, next_state, current_state):

        self.torso_x_position = torso_x_position
        self.torso_z_position = torso_z_position
        self.torso_alpha = torso_alpha
        self.left_hip_alpha = left_hip_alpha
        self.right_hip_alpha = right_hip_alpha
        self.left_knee_alpha = left_knee_alpha
        self.right_knee_alpha = right_knee_alpha
        self.left_ankle_alpha = left_ankle_alpha
        self.right_ankle_alpha = right_ankle_alpha
        self.torso_x_vel = torso_x_vel
        self.torso_z_vel = torso_z_vel
        self.torso_omega = torso_omega
        self.left_hip_omega = left_hip_omega
        self.right_hip_omega = right_hip_omega
        self.left_knee_omega = left_knee_omega
        self.right_knee_omega = right_knee_omega
        self.left_ankle_omega = left_ankle_omega
        self.right_ankle_omega = right_ankle_omega

        self.reward = 0
        self.terminal = 0
        self.current_state = current_state
        self.next_state = next_state

    # ========================================================================================================
    # Computing next step for each state, from current states and actions predicted from actor&critic
    # functions refer to state in the same order as in the init function
    # ========================================================================================================

    def next_txp(self, current, action):
        self.torso_x_position = 0
        return self.torso_x_position

    def next_tzp(self, current, action):
        self.torso_z_position = 0
        return self.torso_z_position

    def next_ta(self, current, action):
        self.torso_alpha = 0
        return self.torso_alpha

    def next_lha(self, current, action):
        self.left_hip_alpha = 0
        return self.left_hip_alpha

    def next_rha(self, current, action):
        self.right_hip_alpha = 0
        return self.right_hip_alpha

    def next_lka(self, current, action):
        self.left_knee_alpha = 0
        return self.left_knee_alpha

    def next_rka(self, current, action):
        self.right_knee_alpha = 0
        return self.right_knee_alpha

    def next_laa(self, current, action):
        self.left_ankle_alpha = 0
        return self.left_ankle_alpha

    def next_raa(self, current, action):
        self.right_ankle_alpha = 0
        return self.right_ankle_alpha

    def next_txv(self, current, action):
        self.torso_x_vel = 0
        return self.torso_x_vel

    def next_tzv(self, current, action):
        self.torso_z_vel = 0
        return self.torso_z_vel

    def next_to(self, current, action):
        self.torso_omega = 0
        return self.torso_omega

    def next_lho(self, current, action):
        self.left_hip_omega = 0
        return self.left_hip_omega

    def next_rho(self, current, action):
        self.right_hip_omega = 0
        return self.right_hip_omega

    def next_lko(self, current, action):
        self.left_knee_omega = 0
        return self.left_knee_omega

    def next_rko(self, current, action):
        self.right_knee_omega = 0
        return self.right_knee_omega

    def next_lao(self, current, action):
        self.left_ankle_omega = 0
        return self.left_ankle_omega

    def next_rao(self, current, action):
        self.right_ankle_omega = 0
        return self.right_ankle_omega

    def current_state(self):
        current_state = [self.torso_x_position, self.torso_z_position, self.torso_alpha,
                         self.left_hip_alpha, self.right_hip_alpha, self.left_knee_alpha,
                         self.right_knee_alpha, self.left_ankle_alpha, self.right_ankle_alpha,
                         self.torso_x_vel, self.torso_z_vel, self.torso_omega,
                         self.left_hip_omega, self.right_hip_omega, self.left_knee_omega,
                         self.right_knee_omega, self.left_ankle_alpha, self.right_ankle_omega]

    def next_states(self, current, action):
        next_state = [self.next_txp(current, action), self.next_tzp(current, action), self.next_ta(current, action),
                      self.next_lha(current, action), self.next_rha(current, action), self.next_lka(current, action),
                      self.next_rka(current, action), self.next_laa(current, action), self.next_raa(current, action),
                      self.next_txv(current, action), self.next_tzv(current, action), self.next_to(current, action),
                      self.next_lho(current, action), self.next_rho(current, action), self.next_lko(current, action),
                      self.next_rko(current, action), self.next_lao(current, action), self.next_rao(current, action)]
        return next_state

    def reset(self):
        self.next_state = [0, 0, -0.101485,
                      0.100951, 0.819996, -0.00146549,
                      -1.27, 4.11e-6, 2.26e-7,
                      0, 0, 0,
                      0, 0, 0,
                      0, 0, 0]
        return self.next_state

    def DoomedToFall(self):
        torsoConstraint = 1
        stanceConstraint = 0.36*m.pi
        torsoHeightConstraint = -0.15
        if m.fabs(self.next_state[2]) > torsoConstraint or m.fabs(self.next_state[7] > stanceConstraint) or \
            m.fabs(self.next_state[8]) > stanceConstraint or self.next_state[5] > 0 or self.next_state[6] > 0 or \
            self.next_state[1] < torsoHeightConstraint:
            return True
        return False

    # def calc_next_state(self, current_state, input):
    #     self.next_state = models.next_states(current_state, input)
    #     return self.next_state

    def calc_reward(self):
        RwDoomedToFall = -75
        RwTime = -1.5
        RwForward = 300
        self.reward += RwTime
        self.reward += RwForward*(self.next_state[0] - self.current_state[0])
        if self.DoomedToFall():
            self.reward += RwDoomedToFall
        return self.reward

    def calc_terminal(self):
        if self.DoomedToFall():
            self.terminal = 2
        else:
            self.terminal = 0
        return self.terminal


mat = sio.loadmat('/home/nicola/SR_NEW/AAA/SR_NEW_CALC/Results_18/1_Torso_xpos/5_1000_30_5_40/model_4.m')
print mat