from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time, pdb

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from alexnet import model
from layers import data_helper

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('gpu', '',
                            """gpu device number""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('ImageSize', 256,
                            """size of image [ size , size ].""")
# depth of image.
DEPTH = 3

def train():
    """Train Character Recognition for a number of steps."""
    GPU = '/gpu:' + str(FLAGS.gpu)
    with tf.Graph().as_default():
        with tf.device(GPU):
            images = tf.placeholder("float", [FLAGS.batch_size,
                                              FLAGS.ImageSize, FLAGS.ImageSize, DEPTH])
            labels = tf.placeholder("int64", [FLAGS.batch_size])

            prep = data_helper.QueueImageData()
            result = prep.distorted_inputs(FLAGS.filename, FLAGS.batch_size, FLAGS.ImageSize)
            # result = recog.inputs()
            img = result.images
            label = result.labels

            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits = model.inference(images)

            # Calculate loss.
            loss = model.loss(logits, labels)

            # Create a saver.
            saver = tf.train.Saver(tf.all_variables())
            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            train_op = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(loss)
            # import pudb; pudb.set_trace()
            # Start running operations on the Graph.
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement))
            # Build an initialization operation to run below.
            init = tf.initialize_all_variables()
            sess.run(init)

            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)

        for step in xrange(FLAGS.max_steps):
                start_time = time.time()
                input_img, input_label = sess.run([img, label])
                _, loss_value = sess.run([train_op, loss], feed_dict={images: input_img, labels: input_label})
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value,
                                        examples_per_sec, sec_per_batch))

                # Save the model checkpoint periodically.
                if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    # tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
