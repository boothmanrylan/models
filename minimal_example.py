import numpy as np
import tensorflow as tf
import sys
from cifar10_model import ResNetCifar10
from cifar10 import Cifar10DataSet

from tensorflow.examples.tutorials.mnist import input_data

n_pseudo_batches =  int(sys.argv[1]) if len(sys.argv) > 1 else 128
actual_batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
iterations =        int(sys.argv[3]) if len(sys.argv) > 3 else 10

tf.set_random_seed(147258)
np.random.seed(123456)

model = ResNetCifar10(2, True, 0.001, 0.1)
dataset = Cifar10DataSet("cifar10_data")

# def model(input):
#   # These initializers ensure that the model will always be
#   # instantiated the same, for comparison.
#   hidden_initializer = tf.constant_initializer(
#                             np.random.uniform(-0.025, 0.025, size=[784,100]))
#   hidden = tf.layers.dense(input, 100, kernel_initializer=hidden_initializer)
#   out_initializer = tf.constant_initializer(
#                             np.random.uniform(-0.025, 0.025, size=[100,10]))
#   return tf.layers.dense(tf.nn.relu(hidden), 10,
#                          kernel_initializer=out_initializer)

inp = tf.placeholder(tf.float32, [None, 32, 32, 3])
targ = tf.placeholder(tf.float32, [None,11])

# Make our model and optimizer and gradients
out = model.forward_pass(inp)
opt = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
loss = tf.losses.mean_squared_error(out, targ)

# standard gradients for a batch
t_vars = tf.trainable_variables()
grads, graph_vars = zip(*opt.compute_gradients(loss, t_vars))

# IMPORTANT: Make sure you call the zero_ops to reset the accumulated gradients
# tl;dr - Add the below section to your code to accumulate gradients
# ------------------------------------------------------------------------
# Define our divisor, used to normalise gradients across pseudo_batches
divisor = tf.Variable(0, trainable=False)
div_fl = tf.to_float(divisor)
zero_divisor = divisor.assign(0)
inc_divisor = divisor.assign(divisor+1)

#   Accumulation ops and variables
# create a copy of all trainable variables with `0` as initial values
accum_grads = [tf.Variable(
                    tf.zeros_like(t_var.initialized_value()), trainable=False)
               for t_var in t_vars]
# create an op to zero all accums vars (and zero the divisor again)
with tf.control_dependencies([zero_divisor]):
  zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_grads]
# Create ops for accumulating the gradient (also adds one to the final divisor)
with tf.control_dependencies([inc_divisor]):
  accum_ops = [accum_grad.assign_add(grad)
               for (accum_grad, grad) in zip(accum_grads, grads)]

# Create op that updates the weights (also divides accumulated gradients by the number of steps)
normalised_accum_grads = [accum_grad/div_fl for (accum_grad) in accum_grads]
# ----------------------------------------------------------------------
train_op = opt.apply_gradients(zip(normalised_accum_grads, graph_vars))

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, seed=764847)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for x in range(iterations):

    iteration_loss = 0
    for y in range(n_pseudo_batches):
      inp_, targ_ = dataset.make_batch(actual_batch_size)
      # inp_, targ_ = mnist.train.next_batch(actual_batch_size)
      _, loss_ = sess.run((accum_ops, loss), {inp: inp_, targ: targ_})
      iteration_loss += loss_
    # To find actual loss, you need to divide by the number of pseudo batches
    print(iteration_loss/n_pseudo_batches)

    sess.run(train_op)
    # vvvv --- VERY IMPORTANT! --- vvvv
    sess.run(zero_ops)
