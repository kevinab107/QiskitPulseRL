# QiskitPulseRL

installation

TODO: Setup python distribution. Currenly the projects setup.py has issues with managing dependancies. Fix it and provide commands for "training", "load best policy".

requires python 3.7 to avoid dependancy issues with tensorflow

pip install -r requirements.txt

if you get any error regarding thread pool you might want to do the following: 
!pip3 uninstall protobuf --yes
!pip3 uninstall python-protobuf --yes
!pip install protobuf 

Alternatively can use the PulseRL-Devo notebook


Description
The  sequence of pulses used for the control of qubits for a unitary operation can vary from the hardware depending on the Rabi frequency, relative detuning, process time, spin relaxation factor and decoherence factor.

The problem statement is as follows

"Given a quantum hardware (with unknown dynamics) can the optimal pulse sequence for a given unitary operation (performed in the give process time) be found using Reinforcement Learning. This will be done with repeated experiments on the hardware until the Algorithm learns the optimal pulse, subject to weak noise. Depending on the spin relaxation and decoherence the algorithm should adjust for the pulses to get the desired final state within an error range. 

Challenges/Questions:

1. How to build the pulse scheduler to have dynamic components which can affect the pulse shape.  The pulse length can be divided into small intervals and apply different magnitude for each interval

2. Benchmark against robust control/optimal control methods where a model of the hardware is used for getting the pulse.

The training of the RL agent can be done with a simulator and existing hardware for a specific unitary operation. (Starting with single qubit operations).

 

Approach

Phase 1: 

Single qubit unitary using qiskit pulse (sigmax operation)

1. Build a pulse schedule with qiskit pulse with width of the pulse as an input.  The schedule is then a function of width (time) of the pulse (with constant amplitude =1), followed by a measurement at the end of the drive. 

3. Use RL to train an agent to learn the "width" of the pulse. Use fidelity (to the expected final state) as the reward. Agent should identify the width of the pulse for a quantum simulator, where an sigmax operation can be implemented. ie The schedule implements a sigmax operator for certain widths
