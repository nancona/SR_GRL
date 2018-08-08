#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on July 2018

@author: nancona
"""

import math as m
import matlab.engine
import numpy as np


class Models(object):

    def __init__(self, torso_x_position, torso_z_position, torso_alpha, left_hip_alpha, right_hip_alpha,
                 left_knee_alpha, right_knee_alpha, left_ankle_alpha, right_ankle_alpha, torso_x_vel, torso_z_vel,
                 torso_omega, left_hip_omega, right_hip_omega, left_knee_omega, right_knee_omega, left_ankle_omega,
                 right_ankle_omega):

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

        self.eng = matlab.engine.start_matlab()
    # ========================================================================================================
    # Computing next step for each state, from current states and actions predicted from actor&critic
    # functions refer to state in the same order as in the init function
    # ========================================================================================================

    def next_txp(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.torso_x_position = self.eng.trsxp_1_model_4(x)
        return self.torso_x_position

    def next_tzp(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.torso_z_position = self.eng.trszp_2_model_3(x)
        return self.torso_z_position

    def next_ta(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.torso_alpha = self.eng.trsa_3_model_1(x)
        return self.torso_alpha

    def next_lha(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.left_hip_alpha = self.eng.lha_4_model_3(x)
        return self.left_hip_alpha

    def next_rha(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.right_hip_alpha = self.eng.rha_5_model_1(x)
        return self.right_hip_alpha

    def next_lka(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.left_knee_alpha = self.eng.lka_6_model_1(x)
        return self.left_knee_alpha

    def next_rka(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.right_knee_alpha = self.eng.rka_7_model_1(x)
        return self.right_knee_alpha

    def next_laa(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.left_ankle_alpha = self.eng.laa_8_model_3(x)
        return self.left_ankle_alpha

    def next_raa(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.right_ankle_alpha = self.eng.raa_9_model_4(x)
        return self.right_ankle_alpha

    def next_txv(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.torso_x_vel = self.eng.trsxv_10_model_1(x)
        return self.torso_x_vel

    def next_tzv(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.torso_z_vel = self.eng.trszv_11_model_1(x)
        return self.torso_z_vel

    def next_to(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.torso_omega = self.eng.trso_12_model_3(x)
        return self.torso_omega

    def next_lho(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.left_hip_omega = self.eng.lho_13_model_4(x)
        return self.left_hip_omega

    def next_rho(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.right_hip_omega = self.eng.rho_14_model_5(x)
        return self.right_hip_omega

    def next_lko(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.left_knee_omega = self.eng.lko_15_model_2(x)
        return self.left_knee_omega

    def next_rko(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.right_knee_omega = self.eng.rko_16_model_5(x)
        return self.right_knee_omega

    def next_lao(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.left_ankle_omega = self.eng.lao_17_model_5(x)
        return self.left_ankle_omega

    def next_rao(self, current, action):
        action = np.ndarray.tolist(action)
        x = current + action
        x = matlab.double(x)
        self.right_ankle_omega = self.eng.rao_18_model_4(x)
        return self.right_ankle_omega

    def current_state(self):
        current_state = [self.torso_x_position, self.torso_z_position, self.torso_alpha,
                         self.left_hip_alpha, self.right_hip_alpha, self.left_knee_alpha,
                         self.right_knee_alpha, self.left_ankle_alpha, self.right_ankle_alpha,
                         self.torso_x_vel, self.torso_z_vel, self.torso_omega,
                         self.left_hip_omega, self.right_hip_omega, self.left_knee_omega,
                         self.right_knee_omega, self.left_ankle_omega, self.right_ankle_omega]
        return current_state

    def next_states(self, current, action):
        next_state = [self.next_txp(current, action), self.next_tzp(current, action), self.next_ta(current, action),
                      self.next_lha(current, action), self.next_rha(current, action), self.next_lka(current, action),
                      self.next_rka(current, action), self.next_laa(current, action), self.next_raa(current, action),
                      self.next_txv(current, action), self.next_tzv(current, action), self.next_to(current, action),
                      self.next_lho(current, action), self.next_rho(current, action), self.next_lko(current, action),
                      self.next_rko(current, action), self.next_lao(current, action), self.next_rao(current, action)]
        return next_state

    def reset(self):
        self.next_state = Models(0, 0, -0.101485,
                                 0.100951, 0.819996, -0.00146549,
                                 -1.27, 4.11e-6, 2.26e-7,
                                 0, 0, 0,
                                 0, 0, 0,
                                 0, 0, 0)
        return self.next_state

    def DoomedToFall(self):
        torsoConstraint = 1
        stanceConstraint = 0.36*m.pi
        torsoHeightConstraint = -0.15
        if m.fabs(self.current_state()[2]) > torsoConstraint or m.fabs(self.current_state()[7] > stanceConstraint) or \
           m.fabs(self.current_state()[8]) > stanceConstraint or self.current_state()[5] > 0 or \
           self.current_state()[6] > 0 or self.current_state()[1] < torsoHeightConstraint:
            return True
        return False

    def calc_reward(self, nextstate, currentstate):
        self.reward = 0
        RwDoomedToFall = -75
        RwTime = -1.5
        RwForward = 300
        self.reward = RwTime
        self.reward += RwForward*(nextstate[0] - currentstate[0])
        if self.DoomedToFall():
            self.reward += RwDoomedToFall
        return self.reward

    def calc_terminal(self):
        if self.DoomedToFall():
            self.terminal = 2
        else:
            self.terminal = 0
        return self.terminal

