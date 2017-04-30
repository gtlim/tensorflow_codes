from scipy.misc import imread
import numpy as np
import tensorflow as tf
from datetime import datetime
FLAGS = tf.app.flags.FLAGS
# Basic compute image mean parameters
tf.app.flags.DEFINE_string('filename','',
                           """Path to the image data textfile directory.""")
tf.app.flags.DEFINE_string('target','',
                           """Target numpy name for image mean""")
import pdb
def compute_image_mean(filename, target):
    f = open(filename, 'r')
    R = 0; G = 0; B = 0
    count = 0
    print(('%s: compute image mean from %s') %(datetime.now(), filename)) 
    for line in f:
        name, _ = line.rstrip().split(' ')
        img = imread(name, mode='RGB')
        mean = np.mean(img, axis=(0,1))
        R += mean[0]
        G += mean[1]
        B += mean[2]
        count+=1
        if count % 1000 == 0:
           format = ('%s: %d image proceed..')
           print(format %(datetime.now(), count)) 
    format = ('%s: %d image proceed..')
    print(format %(datetime.now(), count)) 
    mean = np.around(np.asarray([ R/count, G/count, B/count ]),decimals=2)
    pdb.set_trace() 
    np.save(target, mean)
    format = ('%s: mean:(%.2f, %.2f, %.2f) save as %s')
    print(format %(datetime.now(), mean[0], mean[1], mean[2], target)) 
     
if __name__ == "__main__":
   compute_image_mean(FLAGS.filename,FLAGS.target)
