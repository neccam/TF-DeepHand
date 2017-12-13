#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import tensorflow as tf
import utils
import time
import sys
import platform

# Define Code and Data Path
# code_path: where the classify.py resides
# data_path: where the images folder of eval set resides
code_path = '<ABSOLUTE_PATH_OF_THE_CODE>'
data_path = '<ABSOLUTE_PATH_OF_THE_DATA>'

# For Windows
if platform.system() == 'Windows':
    sys.path.append(code_path+'deephand')
    
from deephand import DeepHand

# Define Batch Size
batch_size = 8;

# input_list_file: a file with relative paths of images that you want to evaluate
input_list_file = code_path + 'input/3359-ph2014-MS-handshape-index.txt'
# mean_path: mean image path
mean_path = code_path + 'input/onemilhands_mean.npy';
# model_path: pretrained weights path
model_path = code_path + 'deephand/deephand_model.npy';

# Define Network
input_node = tf.placeholder(tf.float32, shape=(None, 224, 224, 3));
net = DeepHand({'data': input_node})

# Load Mean Image
mean = np.load(mean_path);
mean = np.transpose(mean, (1, 2, 0));

# Read Image List and Labels
image_paths, labels = utils.read_eval_image_list(input_list_file)

# Get Number of Iterations
num_samples = len(labels);
num_iter = np.int(np.ceil(1.0*num_samples/batch_size));

# Create the storage for image_scores
image_scores = np.zeros((num_iter*batch_size,61))

# Set TF-Session Config Parameters
config=tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True

# Create a session and get posteriors from each frame
with tf.Session(config=config) as sesh:
    
    # Set Device
    with tf.device("/gpu:0"):
        
        # Load Network Weights
        net.load(model_path, sesh)
        
        # Start Timer
        begin = time.time();

        sample_iter = 0;
        
        for i in range(0, num_iter):
            
            batched_input = np.zeros((batch_size, 224, 224, 3));
            
            for j in range(0, batch_size):
                if sample_iter < num_samples:
                    img = utils.read_image(data_path + image_paths[sample_iter]);
                    batched_input[j,:,:,:] = utils.preprocess_image(img, mean);
                    sample_iter += 1;
            
            image_scores[i*batch_size:(i+1)*batch_size,:] = sesh.run(net.get_output(), feed_dict={input_node: batched_input})
            curr = time.time()-begin;
            print("Evaluated {}/{} iterations in {:.2f} seconds - {:.2f} seconds/iteration".format(i+1, num_iter, curr, curr/(i+1)))

        elapsed = time.time()-begin;
        print("Total Evaluation Time: {:.2f} seconds".format(elapsed))


# Only get the valid scores
image_scores = image_scores[0:num_samples,:];

# Get predictions
predictions = np.argmax(image_scores, axis=1);

# Get Accuracy
accuracy = 100.0 * np.sum(predictions == np.int64(np.asarray(labels))) / len(predictions);

# Print out
print("Accruracy on Test Set: {:.4f}".format(accuracy))
