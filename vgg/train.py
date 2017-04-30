from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time, pdb

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from vgg import model
from data_helper import image_preprocessor

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'checkpoints',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('gpu', '',
                            """gpu device number""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('image_size', 256,
                            """size of image [ size , size ].""")
# depth of image.
DEPTH = 3

def train():
    """Train Character Recognition for a number of steps."""
    GPU = '/gpu:' + str(FLAGS.gpu)
    with tf.Graph().as_default():
        with tf.device(GPU):
            prep = image_preprocessor.QueueData(crop_size=224)
            train_result = prep.distorted_inputs(FLAGS.filename, FLAGS.batch_size, FLAGS.image_size)
            validation_result = prep.inputs(FLAGS.filename, FLAGS.batch_size, FLAGS.image_size)
            val_epoch = validation_result.num_examples

            img = train_result.images
            label = train_result.labels
            img_val = validation_result.images
            label_val = validation_result.labels

            images = tf.placeholder("float", [FLAGS.batch_size,
                                              prep.crop_size, prep.crop_size, DEPTH])
            labels = tf.placeholder("int32", [FLAGS.batch_size])
            keep_prob = tf.placeholder("float")

            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits = model.inference(images,keep_prob)
            # Calculate loss.
            loss = model.loss(logits, labels)
            batch_eval = model.eval(logits,labels)
           
            # Create a saver.
            saver = tf.train.Saver(tf.all_variables())
            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            #train_op = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(loss)
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = 0.01
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 0.96, staircase=True)
            #train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
            train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss, global_step=global_step)
            # import pudb; pudb.set_trace()
            # Start running operations on the Graph.
            config = tf.ConfigProto(allow_soft_placement = True,
                                    log_device_placement=FLAGS.log_device_placement)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            # Build an initialization operation to run below.
            init_op = tf.group(tf.initialize_all_variables(),
                   tf.initialize_local_variables()) # when perform sliding producer, error occured.
            sess.run(init_op) 
            #model.load_initial_weights(sess,'alexnet/bvlc_alexnet.npy') 
            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)

            for step in xrange(FLAGS.max_steps):
                
                start_time = time.time()
                input_img, input_label = sess.run([img, label])
                _, loss_value, lr_value = sess.run([train_op, loss, learning_rate], feed_dict={images: input_img, labels: input_label, keep_prob: 0.5})
                duration = time.time() - start_time
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 100 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    format_str = ('%s: step %d, loss = %g lr = %g (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value,
                                        lr_value, examples_per_sec, sec_per_batch))
                    #print(val)
                # Save the model checkpoint periodically.
                # Check model performance periodically using validation set.
                if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    format_str = ('%s: validation..')
                    print(format_str % (datetime.now()))
                    acc = 0.0
                    for step in xrange(int(val_epoch/FLAGS.batch_size)):
                        #val_img, val_label = sess.run([img_val, label_val])
                        val_img, val_label = sess.run([img, label])
                        res, val = sess.run([batch_eval,logits], feed_dict={images: val_img, labels: val_label, keep_prob: 1.0})
                        acc += res
                    format_str = ('%s: validation ends.. acc = %.2f')
                    print(format_str % (datetime.now(), acc/val_epoch))
                        
                    #checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    #saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
