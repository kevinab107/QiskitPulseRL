from environment import QiskitEnv
import numpy as np
from tf_agents.environments import tf_py_environment
from agent import Agent
from qiskit.visualization import plot_bloch_multivector
import matplotlib as plt
from tf_agents.policies import policy_saver
# Learning Parameters
num_iterations = 10  # @param {type:"integer"}
collect_episodes_per_iteration = 250  # @param {type:"integer"}
# for ddpg max repaly = 2
replay_buffer_capacity = 2000  # @param {type:"integer"}

learning_rate = 1e-3  # @param {type:"number"}
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 5  # @param {type:"integer"}
interval_width = 100
num_intervals = 5

agent = Agent(num_iterations,collect_episodes_per_iteration,replay_buffer_capacity,learning_rate,num_eval_episodes,eval_interval)
environment =  QiskitEnv(np.array([0,1]),num_intervals,interval_width)
replay_buffer_capacity = 2000
tf_dumm = tf_py_environment.TFPyEnvironment(environment)
agent_reinforce = agent.get_agent(environment, 'reinforce', "without_noise_trained")
tf_dumm.close()
train_results = agent.train(tf_dumm, agent_reinforce)
env_noisy = QiskitEnv(np.array([0,1]),num_intervals,interval_width)
vector,fid, action = agent.evaluate(agent_reinforce, env_noisy)
policy_dir = "policy"
tf_policy_saver = policy_saver.PolicySaver(agent_reinforce.policy)
tf_policy_saver.save(policy_dir)
print("Fidelity : ", fid)
print("Action", action)
plot_bloch_multivector(vector)
