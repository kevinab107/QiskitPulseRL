import tensorflow as tf 
import numpy as np
from tensorflow.saved_model import load
from tf_agents import policies
from environment import QiskitEnv
import warnings
import logging, sys


def load_best_policy():
  policy_dir = "best_policy"
  # saved_policy = load(policy_dir)
  # print(saved_policy.action())
  converter = tf.lite.TFLiteConverter.from_saved_model(policy_dir, signature_keys=["action"])
  converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
  ]
  tflite_policy = converter.convert()
  with open('policy.tflite', 'wb') as f:
    f.write(tflite_policy)


  interpreter = tf.lite.Interpreter("policy.tflite")

  policy_runner = interpreter.get_signature_runner()

  action = policy_runner(**{
      '0/discount':tf.constant(0.0),
      '0/observation':tf.zeros([1,4]),
      '0/reward':tf.constant(0.0),
      '0/step_type':tf.constant(0)})["action"][0][0]
  
  print("\n\n\n\nPulse length : ",action)
  print("Showing the results of the best policy")
  fid, vector = QiskitEnv.get_state(int(action))
  print("Fidelity : ",fid)
  print("Initial State: ", [1,0])
  print("Final State: ", vector)
  print("\n\n")
  


if __name__ == "__main__":
  logging.disable(sys.maxsize)
  warnings.simplefilter('ignore')
  load_best_policy()
  
