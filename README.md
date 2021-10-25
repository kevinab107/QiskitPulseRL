# QiskitPulseRL

## Description

The sequence of pulses used for the control of qubits for a unitary operation can vary from the hardware depending on the Rabi frequency, relative detuning, process time, spin relaxation factor and decoherence factor.

The problem statement is as follows:

"Given a quantum hardware (with unknown dynamics) can the optimal pulse sequence for a given unitary operation (performed in the give process time) be found using Reinforcement Learning. This will be done with repeated experiments on the hardware until the Algorithm learns the optimal pulse, subject to weak noise. Depending on the spin relaxation and decoherence the algorithm should adjust for the pulses to get the desired final state within an error range."


1. How to build the pulse scheduler to have dynamic components which can affect the pulse shape. The pulse length can be divided into small intervals and apply different magnitude for each interval

2. Benchmark against robust control/optimal control methods where a model of the hardware is used for getting the pulse.

The training of the RL agent can be done with a simulator and existing hardware for a specific unitary operation. (Starting with single qubit operations).

## Installation
**Use the PulseRL-Devo Jupyter notebook** or alternatively install by running
```
pip install -r requirements.txt
pip install .
```
Requires python 3.7 to avoid dependency issues with tensorflow.

If you get any error regarding thread pool you might want to do the following: 

```
!pip3 uninstall protobuf --yes
!pip3 uninstall python-protobuf --yes
!pip install protobuf 
```

TODO: Currently the projects setup.py has issues with managing dependencies. Fix and provide commands for "training", "load best policy".

## Prior work
based on existing project, see: https://github.com/kevinab107/QiskitPulseRL/commit/05956333c201db050b1e8ec1c1c9890c4ab6c17d
