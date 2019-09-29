#!/usr/bin/env python

from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf
import utils
import time
import sys
import platform


def main(param):

    code_path = param.code_path
    data_path = param.data_path
    batch_size = param.batch_size

    # For Windows
    if platform.system() == "Windows":
        sys.path.append(os.path.join(code_path, "deephand"))
    from deephand import DeepHand

    # input_list_file: a file with relative paths of images that you want to evaluate
    input_list_file = os.path.join(
        code_path, "input/3359-ph2014-MS-handshape-index.txt"
    )
    # mean_path: mean image path
    mean_path = os.path.join(code_path, "input/onemilhands_mean.npy")
    # model_path: pretrained weights path
    model_path = os.path.join(code_path, "deephand/deephand_model.npy")

    # Define Network
    input_node = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    net = DeepHand({"data": input_node})

    # Load Mean Image
    mean = np.load(mean_path)
    mean = np.transpose(mean, (1, 2, 0))

    # Read Image List and Labels
    image_paths, labels = utils.read_eval_image_list(input_list_file)

    # Get Number of Iterations
    num_samples = len(labels)
    num_iter = np.int(np.ceil(1.0 * num_samples / batch_size))

    # Create the storage for image_scores
    image_scores = np.zeros((num_iter * batch_size, 61))

    # Set TF-Session Config Parameters
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    # Create a session and get posteriors from each frame
    with tf.Session(config=config) as sesh:

        # Set Device
        with tf.device("/gpu:0"):

            # Load Network Weights
            net.load(model_path, sesh)
            sesh.run(
                net.get_output(),
                feed_dict={input_node: np.zeros((batch_size, 224, 224, 3))},
            )

            # Start Timer
            begin = time.time()

            sample_iter = 0

            for i in range(0, num_iter):

                batched_input = np.zeros((batch_size, 224, 224, 3))

                for j in range(0, batch_size):
                    if sample_iter < num_samples:
                        img = utils.read_image(
                            os.path.join(data_path + image_paths[sample_iter])
                        )
                        batched_input[j, :, :, :] = utils.preprocess_image(img, mean)
                        sample_iter += 1

                image_scores[i * batch_size : (i + 1) * batch_size, :] = sesh.run(
                    net.get_output(), feed_dict={input_node: batched_input}
                )
                curr = time.time() - begin
                print(
                    "Evaluated {}/{} iterations in {:.2f} seconds - {:.2f} seconds/iteration".format(
                        i + 1, num_iter, curr, curr / (i + 1)
                    )
                )

            elapsed = time.time() - begin
            print("Total Evaluation Time: {:.2f} seconds".format(elapsed))

    # Only get the valid scores
    image_scores = image_scores[0:num_samples, :]

    # Get predictions
    predictions = np.argmax(image_scores, axis=1)

    # Get Accuracy
    accuracy = (
        100.0 * np.sum(predictions == np.int64(np.asarray(labels))) / len(predictions)
    )

    # Print out
    print("Accuracy on Test Set: {:.4f}".format(accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--code_path",
        type=str,
        default="<ABSOLUTE_PATH_OF_THE_CODE>",
        help="Directory where the evaluate.py and deephand folder resides",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="<ABSOLUTE_PATH_OF_THE_DATA>",
        help="Directory of the images folder of one-million-hands eval set.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference"
    )

    main(param=parser.parse_args())
