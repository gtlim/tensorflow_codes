from common import (
    variable_on_cpu, variable_with_weight_decay,
    batch_flatten
)
import tensorflow as tf

def repeat_layer(repetitions, inputs, op, name , *args, **kwargs):
    tower = inputs
    for idx in range(repetitions):
      new_name = name + '_' + str(idx)
      tower = op(new_name, tower, *args, **kwargs)
    return tower

def conv(_input, kernel, biases, c_o, s_h, s_w, padding, group):
    """
        Convolution Layer
        Params: _input: tensor
                kernel:
    """
    c_i = _input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(_input, kernel)
    else:
        input_groups = tf.split(3, group, _input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)

    if biases is not None:
        return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    else:
        return conv


def conv_2d(name, _input, out_channel, kernel_shape, stride,
            padding, group, stddev, wd, bias, activation=True):

    k_h = k_w = kernel_shape
    s_h = s_w = stride
    c_o = out_channel 
    in_shape = _input.get_shape().as_list()
    in_channel = in_shape[-1] / group
    with tf.variable_scope(name):
        kernel = variable_with_weight_decay('weights', shape=[k_h, k_w, in_channel, c_o],
                                            stddev=stddev, wd=wd)
        if bias is not None:
            biases = variable_on_cpu('biases', [c_o], tf.constant_initializer(bias))
        else:
            biases = None
    if activation:
        return tf.nn.relu(conv(_input, kernel, biases, c_o, s_h, s_w, padding, group))
    else:
        return conv(_input, kernel, biases, c_o, s_h, s_w, padding, group)


def max_pooling(_input, kernel_shape, stride, padding):
    k_h = k_w = kernel_shape
    s_h = s_w = stride
    return tf.nn.max_pool(_input, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


def maxout(_input, group):
    c_i = _input.get_shape()[-1]
    assert c_i % group == 0
    maxout_g2 = lambda i, k: tf.maximum(i, k)
    maxout_g4 = lambda i, j, k, l: tf.maximum(tf.maximum(i, j), tf.maximum(k, l))

    if group == 2:
        input_groups = tf.split(3, c_i, _input)
        output_groups = [maxout_g2(input_groups[i],input_groups[i+1]) for i in xrange(0, c_i, group)]
        __maxout = tf.concat(3, output_groups)
    if group == 4:
        input_groups = tf.split(3, c_i, _input)
        output_groups = [maxout_g4(input_groups[i], input_groups[i+1], input_groups[i+2], input_groups[i+3]) for i in xrange(0, c_i, group)]
        __maxout = tf.concat(3, output_groups)
    else:
        raise
    return tf.reshape(__maxout, [-1] + __maxout.get_shape().as_list()[1:])


def lrn(_input, radius, alpha, beta, bias):
    return tf.nn.local_response_normalization(_input, depth_radius=radius,
                                                 alpha=alpha,
                                                 beta=beta,
                                                 bias=bias)


def batch_norm(bn_name,scale_name,x):
    shape = x.get_shape().as_list()
    assert len(shape) in [2, 4]
    n_out = shape[-1]  # channel
    with tf.variable_scope(bn_name) as scope:
        gamma = variable_on_cpu('gamma', [n_out], tf.constant_initializer(1.0))

    with tf.variable_scope(scale_name) as scope:
        beta = variable_on_cpu('beta', [n_out],None)
    # calculate the mean and variance of x
    if len(shape) == 2:
        batch_mean, batch_var = tf.nn.moments(x, [0], keep_dims=False)
    else:
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=False)

    return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-12)


def fc(name, x, out_dim, stddev, wd, bias, active=True):
     x = batch_flatten(x)
     in_dim = x.get_shape().as_list()[1]
     with tf.variable_scope(name) as scope:
        weights = variable_with_weight_decay('weights', shape=[in_dim, out_dim],
                                         stddev=stddev, wd=wd)
        biases = variable_on_cpu('biases', [out_dim], tf.constant_initializer(bias))
     if active:
        return  tf.nn.relu_layer(x, weights, biases)
     else:
        return  tf.add(tf.matmul(x, weights), biases)
       

def dropout(x,keep_prob):
    return tf.nn.dropout(x,keep_prob)


def bottleneck_unit(name, x, out_channel1, out_channel2, down_stride=False):
    in_channel = x.get_shape().as_list()[3]
    if down_stride:
        first_stride = 2
    else:
        first_stride = 1

    with tf.variable_scope('res%s' % name):
        if in_channel == out_channel2:
            b1 = x
        else:
            with tf.variable_scope('branch1'):
                print('res%s_branch1' % name)
                print(x.get_shape())
                b1 = conv_2d('res%s_branch1' % name, x, out_channel2, kernel_shape=1, stride=first_stride,
                             padding='SAME', group=1, stddev=0.001, wd=0,
                             bias=None, activation=False)
                b1 = batch_norm('bn%s_branch1' % name, 'scale%s_branch2a' % name, b1)

        with tf.variable_scope('branch2a'):
            print('res%s_branch2a' % name)
            print(x.get_shape())

            b2 = conv_2d('res%s_branch2a' % name, x, out_channel1, kernel_shape=1, stride=first_stride,
                         padding='SAME', group=1, stddev=0.001, wd=0,
                         bias=None, activation=False)
            b2 = batch_norm('bn%s_branch2a' % name, 'scale%s_branch2a' % name, b2)
            b2 = tf.nn.relu(b2, name='relu')

        with tf.variable_scope('branch2b'):
            b2 = conv_2d('res%s_branch2b' % name, b2, out_channel1, kernel_shape=3, stride=1,
                         padding='SAME', group=1, stddev=0.001, wd=0,
                         bias=None, activation=False)
            b2 = batch_norm('bn%s_branch2b' % name, 'scale%s_branch2b' % name, b2)
            b2 = tf.nn.relu(b2, name='relu')

        with tf.variable_scope('branch2c'):
            b2 = conv_2d('res%s_branch2b' % name, b2, out_channel2, kernel_shape=1, stride=1,
                         padding='SAME', group=1, stddev=0.001, wd=0,
                         bias=None, activation=False)
            b2 = batch_norm('bn%s_branch2c' % name, 'scale%s_branch2c' % name, b2)
        x = b1 + b2
        return tf.nn.relu(x, name='relu')


def _bottleneck_block(name, x, num_units, out_chan1, out_chan2, down_stride, use_letters):
    for i in range(0, num_units):
        ds = (i == 0 and down_stride)
        if i == 0:
            unit_name = '%sa' % name
        elif use_letters:
            unit_name = '%s%c' % (name, ord('a') + i)
        else:
            unit_name = '%sb%d' % (name, i)

        x = bottleneck_unit(unit_name, x, out_chan1, out_chan2, ds)
    return x


def bottleneck_block(name, x, num_units, out_chan1, out_chan2, down_stride):
    return _bottleneck_block(name, x, num_units, out_chan1, out_chan2, down_stride, use_letters=False)


def bottleneck_block_letters(name, x, num_units, out_chan1, out_chan2, down_stride):
    return _bottleneck_block(name, x, num_units, out_chan1, out_chan2, down_stride, use_letters=True)

