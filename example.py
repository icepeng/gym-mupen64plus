#!/bin/python
import gym, gym_mupen64plus
import subprocess

env = gym.make('Mario-Kart-Luigi-Raceway-v0')

while True:
    env.reset()

    for i in range(88):
        (obs, rew, end, info) = env.step([0, 0, 0, 0, 0]) # NOOP until green light

    for i in range(100):
        (obs, rew, end, info) = env.step([0, 0, 1, 0, 0]) # Drive straight
