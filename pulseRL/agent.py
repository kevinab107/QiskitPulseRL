from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import tf_agents
import math
import matplotlib as plt

import tensorflow as tf

import tf_agents
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from itertools import combinations
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
from tf_agents.networks import actor_distribution_network
from tf_agents.trajectories import trajectory
from tf_agents.agents.reinforce import reinforce_agent
import numpy
from tf_agents.policies.policy_saver import PolicySaver
import tensorflow_addons as tfa
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from environment import QiskitEnv


class Agent:
    """docstring for ClassName"""

    def __init__(
        self,
        num_iterations,
        collect_episodes_per_iteration,
        replay_buffer_capacity,
        learning_rate,
        num_eval_episodes,
        eval_interval,
        num_intervals,
        interval_length,
    ):
        # Learning Parameters
        self.num_iterations = num_iterations  # @param {type:"integer"}
        self.collect_episodes_per_iteration = (
            collect_episodes_per_iteration  # @param {type:"integer"}
        )
        # for ddpg max repaly = 2
        self.replay_buffer_capacity = replay_buffer_capacity  # @param {type:"integer"}

        self.learning_rate = learning_rate  # @param {type:"number"}
        self.num_eval_episodes = num_eval_episodes  # @param {type:"integer"}
        self.eval_interval = eval_interval  # @param {type:"integer"}
        self.num_intervals = num_intervals
        self.interval_length = interval_length

    def get_reinforce_agent(self, spin_py_environment, name):

        # Get the train env from the python environment QiskitEnv
        train_env = tf_py_environment.TFPyEnvironment(spin_py_environment)
        # Define the learning neural network
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            train_env.observation_spec(), train_env.action_spec()
        )
        # Define the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Get the reinforcment Learning Agent
        name = name + "_reinforce"
        tf_agent = reinforce_agent.ReinforceAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            actor_network=actor_net,
            optimizer=optimizer,
            normalize_returns=True,
            name=name,
        )

        tf_agent.initialize()

        return tf_agent

    def get_agent(self, env, agent_type, name):
        return self.get_reinforce_agent(env, name)

    def compute_avg_return(self, environment, policy, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
                print(_, time_step.observation[0], time_step.reward)
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def collect_episode(self, environment, replay_buffer, policy, num_episodes):
        """
        In an iteration multiple episodes are collected together and a trajectory is built out of it.
        Later these trajectory is used for learning. Trajectory is added to a replay buffer.
        """

        episode_counter = 0
        environment.reset()

        while episode_counter < num_episodes:
            time_step = environment.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)

            # Add trajectory to the replay buffer
            replay_buffer.add_batch(traj)

            if traj.is_boundary():
                episode_counter += 1

    def train(self, dummy_env, tf_agent):

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=dummy_env.batch_size,
            max_length=self.replay_buffer_capacity,
        )

        tf_agent.train = common.function(tf_agent.train)

        # Reset the train step
        tf_agent.train_step_counter.assign(0)

        avg_return = self.compute_avg_return(dummy_env, tf_agent.policy, 10)
        returns = [avg_return]

        for _ in range(self.num_iterations):

            train_env = QiskitEnv.get_tf_environment(
                self.num_intervals, self.interval_length
            )
            self.collect_episode(
                train_env,
                replay_buffer,
                tf_agent.collect_policy,
                self.collect_episodes_per_iteration,
            )

            # Use data from the buffer and update the agent's network.
            experience = replay_buffer.gather_all()
            train_loss = tf_agent.train(experience)
            replay_buffer.clear()

            step = tf_agent.train_step_counter.numpy()

            if step % 100 == 0:
                print("step = {0}: loss = {1}".format(step, train_loss.loss))

            if step % self.eval_interval == 0:

                eval_env = QiskitEnv.get_tf_environment(self.num_intervals, self.interval_length)
                avg_return = self.compute_avg_return(
                    eval_env, tf_agent.policy, self.num_eval_episodes
                )
                print("step = {0}: Average Return = {1}".format(step, avg_return))
                returns.append(avg_return)
        return (step, train_loss, returns)

    def plot_training_return(self):
        steps = range(0, self.num_iterations + 1, self.eval_interval)
        plt.plot(steps, returns)
        plt.ylabel("Average Return")
        plt.xlabel("Step")
        plt.ylim(top=2)

    def evaluate(self, tf_agent, eval_py_env):
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
        num_episodes = 1
        fidelity = []
        actions = []
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            while not time_step.is_last():
                action_step = tf_agent.policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                actions.append(action_step.action)
        fidelity, state, pulse_prog = eval_py_env.get_state(actions)
        return state, fidelity, actions, pulse_prog
