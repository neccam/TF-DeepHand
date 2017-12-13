# TF-DeepHand

This is a TensorFlow implementation of Koller et al.'s ["Deep Hand: How to Train a CNN on 1 Million Hand Images When Your Data is Continuous and Weakly Labelled"](http://www-i6.informatik.rwth-aachen.de/~koller/1miohands/) (CVPR'16) paper.

The pretrained caffe model is converted using ethereon's [Caffe to Tensorflow](https://github.com/ethereon/caffe-tensorflow) tool.

Tested with:

* Windows 10   - TensorFlow 1.4.0 - Python 3.5 - GTX 960M
* Ubuntu 14.04 - TensorFlow 1.2.1 - Python 2.7 - Titan X
## Evaluation
To evaluate the model on [One-Million-Hands](https://www-i6.informatik.rwth-aachen.de/~koller/1miohands-data/) dataset's test set:

1. Download and extract this repository and the test data to desired locations.
2. Change `code_path`  and `data_path` accordingly in `evaluation.py` script.
3. Download the [deephand_model.npy](http://cihancamgoz.com/files/tf-deephand/deephand_model.npy) model weights and place it in the `deephand` folder
4. Run `python evaluation.py`

Once the evaluation is done you should see:

    Accruracy on Test Set: 85.4421

This code is set to use the first GPU of your machine. You can easily change it to use any other GPU/CPU by changing the following line in `evaluation.py`:

    with tf.device("/gpu:0"):

## Reference
Please cite the Deep Hand paper if you use this code in your research:

    @inproceedings{koller16:deephand,
      author = {Oscar Koller and Hermann Ney and Richard Bowden},
      title = {Deep Hand: How to Train a CNN on 1 Million Hand Images When Your Data Is Continuous and Weakly Labelled},
      booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2016}
    }
