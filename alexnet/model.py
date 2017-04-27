"""
Builds the Alexnet

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

weight_decay=0.005

def inference(images, is_training=True):
    """Build the Character Recognition model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    conv1 = layers.conv_2d('conv1', images, 96, 11, 4, padding='VALID', group=1, stddev=0.01, wd=weight_decay, bias=0)

    # lrn1
    # lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn1 = layers.lrn(conv1, radius, alpha, beta, bias)

    # maxpool1
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    maxpool1 = layers.max_pooling(lrn1, 3, 2, padding='VALID')

    # conv2
    conv2 = layers.conv_2d('conv2', maxpool1, 256, 5, 1, padding='SAME', group=2, stddev=0.01, wd=weight_decay, bias=0.1)

    # lrn2
    # lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2;
    alpha = 2e-05;
    beta = 0.75;
    bias = 1.0
    lrn2 = layers.lrn(conv2, radius, alpha, beta, bias)

    # maxpool2
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    maxpool2 = layers.max_pooling(lrn2, 3, 2, padding='VALID')

    # conv3
    conv3 = layers.conv_2d('conv3', maxpool2, 384, 3, 1, padding='SAME', group=1, stddev=0.01, wd=weight_decay, bias=0.1)

    # conv4
    conv4 = layers.conv_2d('conv4', conv3, 384, 3, 1, padding='SAME', group=2, stddev=0.01, wd=weight_decay, bias=0.1)

    # conv5
    conv5 = layers.conv_2d('conv5', conv4, 256, 3, 1, padding='SAME', group=2, stddev=0.01, wd=weight_decay, bias=0.1)

    # maxpool5
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    maxpool5 = layers.max_pooling(conv5, 3, 2, padding='VALID')

    # Fc6
    # fc(4096, name='fc6')
    fc6 = layers.fc('fc6', maxpool5, 4096, stddev=0.005, wd=weight_decay, bias=0.1)
    fc6 = layers.dropout(fc6, 0.5, is_training)

    # Fc7
    # fc(4096, name='fc7')
    fc7 = layers.fc('fc7', fc6, 4096, stddev=0.001, wd=weight_decay, bias=0.1)
    fc7 = layers.dropout(fc7, 0.5, is_training)

    # softmax, i.e. softmax(WX + b) fc8
    # fc(1000, relu=False, name='fc8')
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
    return top_k_op 
        
