#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on July 2018

@author: nancona
"""

import math as m
from models import Models

class Step(object):

    def __init__(self, current_state, next_state=None):
        self.reward = 0
        self.terminal = 0
        self.current_state = current_state
        self.next_state  = next_state

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

