import warnings
import logging, sys

import numpy as np

import tensorflow as tf
from tensorflow.saved_model import load
from tf_agents import policies

from pulseRL.environment import QiskitEnv
from tf_agents.environments import tf_py_environment


def evaluate(policy, eval_py_env):
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    num_episodes = 1
    fidelity = []
    actions = []
    for _ in range(num_episodes):
        time_step = eval_env.reset()
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            actions.append(action_step.action)
    fidelity, state, pulse_prog = eval_py_env.get_state(actions)
    return state, fidelity, actions, pulse_prog


env_test = QiskitEnv(np.array([0, 1]), 5, 100)
policy_dir = "best_policy"
saved_policy = load(policy_dir)
state, fidelity, actions, pulse_prog = evaluate(saved_policy, env_test)
print("Showing the results of the best policy")
print("Fidelity : ", fidelity)
print("Initial State: ", [1, 0])
print("Final State: ", state)
print("\n\n")
pulse_prog.draw()
