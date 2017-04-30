"""
 Builds the VGG-16 net

 Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from layers import layers
import tensorflow as tf
import numpy as np
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 256,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('filename', '',
                           """Path to the image data textfile directory.""")
tf.app.flags.DEFINE_integer('num_classes',0 ,
                            """Number of classes.""")

# (self.feed('data')
#         .conv(9, 9, 96, 4, 4, padding='VALID', name='conv1') 11 -> 9
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

SKIP_LAYER = ['fc8']
def load_initial_weights(session, weights_path):

  # Load the weights into memory
  weights_dict = np.load(weights_path, encoding = 'bytes').item()

  # Loop over all layer names stored in the weights dict
  for op_name in weights_dict:

    # Check if the layer is one of the layers that should be reinitialized
    # print(op_name)
    if op_name not in SKIP_LAYER:

      with tf.variable_scope(op_name, reuse = True):

        # Loop over list of weights/biases and assign them to their corresponding tf variable
        for data in weights_dict[op_name]:

          # Biases
          if len(data.shape) == 1:

            var = tf.get_variable('biases', trainable = False)
            session.run(var.assign(data))

          # Weights
          else:

            var = tf.get_variable('weights', trainable = False)
            session.run(var.assign(data))



weight_decay=0.0005 # default weight decay
def inference(images, keep_prob):
    """Build the Character Recognition model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # conv1
    conv1 = layers.repeat_layer(2, images, layers.conv_2d, 'conv1', 64, 3, 1, padding='SAME', group=1, stddev=0.01, wd=weight_decay, bias=0 )
    # maxpool1
    maxpool1 = layers.max_pooling(conv1, 2, 2, padding='VALID')

    # conv2
    conv2 = layers.repeat_layer(2, maxpool1, layers.conv_2d, 'conv2', 128, 3, 1, padding='SAME', group=1, stddev=0.01, wd=weight_decay, bias=0 )
    # maxpool2
    maxpool2 = layers.max_pooling(conv2, 2, 2, padding='VALID')

    # conv3
    conv3 = layers.repeat_layer(3, maxpool2, layers.conv_2d, 'conv3', 256, 3, 1, padding='SAME', group=1, stddev=0.01, wd=weight_decay, bias=0 )
    # maxpool3
    maxpool3 = layers.max_pooling(conv3, 2, 2, padding='VALID')

    # conv4
    conv4 = layers.repeat_layer(3, maxpool3, layers.conv_2d, 'conv4', 512, 3, 1, padding='SAME', group=1, stddev=0.01, wd=weight_decay, bias=0 )
    # maxpool4
    maxpool4 = layers.max_pooling(conv4, 2, 2, padding='VALID')

    # conv5
    conv5 = layers.repeat_layer(3, maxpool4, layers.conv_2d, 'conv5', 512, 3, 1, padding='SAME', group=1, stddev=0.01, wd=weight_decay, bias=0 )
    # maxpool5
    maxpool5 = layers.max_pooling(conv5, 2, 2, padding='VALID')

    # fc(4096, name='fc6')
    fc6 = layers.fc('fc6', maxpool5, 4096, stddev=0.01, wd=weight_decay, bias=0.1)
    fc6 = layers.dropout(fc6, keep_prob)

    # Fc7
    # fc(4096, name='fc7')
    fc7 = layers.fc('fc7', fc6, 4096, stddev=0.01, wd=weight_decay, bias=0.1)
    fc7 = layers.dropout(fc7, keep_prob)

    # softmax, i.e. softmax(WX + b) fc8
    # fc(NUMCLASSES, relu=False, name='fc8')
    softmax_linear = layers.fc('fc8', fc7, FLAGS.num_classes,
                                stddev=0.01, wd=weight_decay, bias=0.1, active=False)
    return softmax_linear


def loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def eval(logits, labels):
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    res = tf.reduce_sum(tf.cast(top_k_op, tf.float32))
    return res
        
