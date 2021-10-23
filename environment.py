
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math 
from qiskit.quantum_info import state_fidelity
from qiskit.pulse import DriveChannel
from qiskit.compiler import assemble
from qiskit.qobj.utils import MeasLevel, MeasReturnType

# The pulse simulator
from qiskit.providers.aer import PulseSimulator
from qiskit import pulse

# Object for representing physical models
from qiskit.providers.aer.pulse import PulseSystemModel

# Mock Armonk backend
from qiskit.test.mock.backends.armonk.fake_armonk import FakeArmonk

import abc
import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
import math

from tf_agents.environments import validate_py_environment



class QiskitEnv(py_environment.PyEnvironment):
    """ Environemnt which make use of Qiskit Pulse Simulator and pulse builder to simulate 
    the dynamics of a qubit under the influence of a pulse. The RL agent interact with this 
    environmentthrough action defined as pulse lenght. Here a constant pulse of amplitude 1 
    is used and applied for a time "pulse width". "pulse width" is the action that the agent 
    takes here. The agent observes the state obtained with the action along with the Fidelity 
    to the expected final state"""

    def __init__(self,
                 initial_state, max_stamp):

        #action spec which is the shape of the action. Here it is (1,) ie the pulse length                 
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int, minimum=10, maximum=1000, name='action')
        #Observation spec which is the shape of the observation. It is of the form [real part of |0>, imag part of |0>, real part of |1>, imag part of |1>]
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,), dtype=np.float32, minimum=0,maximum=1, name='observation')
        self._state = np.array([1,0])  # | 0 > 
        self._episode_ended = False
        #Closeness to the final state defined for this program as [0,1]
        self.fidelity = 0
        #Time stamp 
        self.time_stamp = 0
        self.max_fidelity = 0
        #intial state is [1,0]
        self.initial_state = initial_state
        #Maximum time steps
        self.max_stamp = max_stamp


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        """
        reset the state. Tesnforflow provides the state restart()
        """
        self._episode_ended = False
        self.fidelity = 0
        self.time_stamp = 0
        self.max_fidelity = 0
        #self.gamma = [random.uniform(0, 1), random.uniform(0, 1)]
        return ts.restart(np.array([0,0,1,0], dtype=np.float32))



    def _step(self, action):
        """
        Result of interaction by agent with the environment.
        action is the pulse length
        Returns one of the the tensorflow state which has the observation and 
        reward information corresponding to the action
                      - termination (If the interaction has stopped due to max timesteps)
                      - transition (Transition from one state to another)
                      
        """

        #Set episode ended to True if max time steps has exceeded
        # Terminate with reset in that case
        self.time_stamp += 1
        if self.time_stamp > self.max_stamp:
            self._episode_ended = True
        else:
            self._episode_ended = False

        if self._episode_ended:
            self.max_fidelity = 0
            return ts.termination(np.array([0,0,1,0], dtype=np.float32), 0)

        #Get the new state and fidelity
        new_fidelity, state = self.get_transition_fidelity(action)

        #reward = 2*new_fidelity - self.fidelity - self.max_fidelity
        #reward = reward if reward > 0 else 0
        #reward = new_fidelity
        reward = new_fidelity

        self.fidelity = new_fidelity
        self.max_fidelity = new_fidelity if new_fidelity > self.max_fidelity else self.max_fidelity

        observation = [state[0].real, state[0].imag,state[1].real, state[1].imag ]

        if (self.time_stamp == self.max_stamp):
            self.max_fidelity = 0
            self.fidelity = 0
            #self.gamma = [random.uniform(0, 1), random.uniform(0, 1)]
            return ts.termination(np.array(observation, dtype=np.float32), reward=reward)

        else:
            return ts.transition(
                np.array(observation, dtype=np.float32), reward=reward, discount=0.9)

  

    def get_transition_fidelity(self,width):
        """
        Build the pulse based on the action and invoke a IBM Q backend and run the experiment.Simulator is used 
        here.  
        """
        armonk_backend = FakeArmonk()
        freq_est = 4.97e9
        drive_est = 6.35e7
        armonk_backend.defaults().qubit_freq_est = [freq_est]
        #Define the hamiltonian to avoid randomness
        armonk_backend.configuration().hamiltonian['h_str']= ['wq0*0.5*(I0-Z0)', 'omegad0*X0||D0']
        armonk_backend.configuration().hamiltonian['vars'] = {'wq0': 2 * np.pi * freq_est, 'omegad0': drive_est}
        armonk_backend.configuration().hamiltonian['qub'] = {'0': 2}
        armonk_backend.configuration().dt = 2.2222222222222221e-10
        armonk_model = PulseSystemModel.from_backend(armonk_backend)

        width = int(width)
        #build the pulse 
        with pulse.build(name='pulse_programming_in', backend =  armonk_backend ) as pulse_prog:

                dc = pulse.DriveChannel(0)
                ac = pulse.acquire_channel(0)

                pulse.play([1]*width +[0]*width  , dc)
                pulse.delay(width + 10, ac)
                mem_slot = pulse.measure(0)

        #Simulate the pulse 
        backend_sim = PulseSimulator(system_model=armonk_model)
        qobj = assemble(pulse_prog,
                                backend=backend_sim,
                                meas_return='avg',
                                shots=512)
        sim_result = backend_sim.run(qobj).result()
        vector = sim_result.get_statevector()
        fid = state_fidelity(np.array([0,1]), vector)
        return fid,vector

    # if __name__ == "__main__":
    #     environment =  QiskitEnv(np.array([0,1]),100)
    #     validate_py_environment(environment, episodes=5)
    @staticmethod
    def get_tf_environment(max_step):
        """Return the tenforflow environemnt of the python  environment """
        py_env = QiskitEnv(np.array([1,0]),2)
        tf_env = tf_py_environment.TFPyEnvironment(py_env)
        return tf_env

    @staticmethod
    def get_state(width):
            
            #intiial state is fixed as of now to |0>
            armonk_backend = FakeArmonk()
            freq_est = 4.97e9
            drive_est = 6.35e7
            armonk_backend.defaults().qubit_freq_est = [freq_est]
            armonk_backend.configuration().hamiltonian['h_str']= ['wq0*0.5*(I0-Z0)', 'omegad0*X0||D0']
            armonk_backend.configuration().hamiltonian['vars'] = {'wq0': 2 * np.pi * freq_est, 'omegad0': drive_est}
            armonk_backend.configuration().hamiltonian['qub'] = {'0': 2}
            armonk_backend.configuration().dt = 2.2222222222222221e-10
            armonk_model = PulseSystemModel.from_backend(armonk_backend)

            width = int(width)
            with pulse.build(name='pulse_programming_in', backend =  armonk_backend ) as pulse_prog:

                    dc = pulse.DriveChannel(0)
                    ac = pulse.acquire_channel(0)

                    pulse.play([1]*width +[0]*width  , dc)
                    pulse.delay(width + 10, ac)
                    mem_slot = pulse.measure(0)
        
            backend_sim = PulseSimulator(system_model=armonk_model)
            qobj = assemble(pulse_prog,
                                    backend=backend_sim,
                                    meas_return='avg',
                                    shots=512)
            sim_result = backend_sim.run(qobj).result()
            vector = sim_result.get_statevector()
            fid = state_fidelity(np.array([0,1]), vector)
            return fid, vector

