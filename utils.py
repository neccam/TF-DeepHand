import cv2
import numpy as np

__all__ = ["read_image", "preprocess_image", "read_eval_image_list", "read_image_list"]

def read_image(image_path):
    img = cv2.imread(image_path);
    return img

def preprocess_image(img, mean):
    img = cv2.resize(img, (227, 227));              
    img = img - mean;
    img = img[1:225,1:225,:];
    img = np.expand_dims(img, axis=0)    
    return img

def read_eval_image_list(input_file):
    
    image_paths = [];
    labels = [];
    
    with open(input_file) as f:
        for line in f:
            ip, l = line.strip().split(' ');
            image_paths.append(ip);
            labels.append(l);
    
    return image_paths, labels


def read_image_list(input_file):
    
    image_paths = [];
    
    with open(input_file) as f:
        for line in f:
            image_paths.append(line.strip())
            
    return image_paths
