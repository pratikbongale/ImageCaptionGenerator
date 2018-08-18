# Our goal is to use transfer learning by extracting the image features using already trained VGG model
# and store them in a pickle file.
# References:
# https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
# https://pillow.readthedocs.io/en/5.1.x/handbook/image-file-formats.html

import os
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def extract_features(dir):

    # 1. load model
    # 2. get rid of the last layer
    # 3. extract features for each image in given dir
    # 4. store as a pickle

    #  check if directory exixts
    if not os.path.exists(dir):
        print('Unable to find images directory')
        return

    model = VGG16()
    model.layers.pop()

    print('Summary of the model')
    print(model.summary())

    features_dict = dict()

    for img_name in os.listdir(dir):
        fpath = os.path.join(dir, img_name)
        pil_img = load_img(fpath)   # load an img in Python Imaging Library(PIL) format
        img_arr = img_to_array(pil_img)     # 3D numpy array (x, y, layer(RGB))
        img_shape = img_arr.shape
        flat_img_arr = img_arr.reshape((1, img_shape[0], img_shape[1], img_shape[2]))
        image = preprocess_input(flat_img_arr)  # preprocessing powered for imagenet
        features = model.predict(image, verbose=0)
        image_id = img_name.split('.')[0]
        features_dict[image_id] = features

    return features_dict

if __name__ == '__main__':

    # test 1
    dir_abs = '/home/pratik/Desktop/Projects/ImageCaptionGenerator/Dataset/Flickr8k_Dataset/Flicker8k_Dataset'
    dir_rel = 'Dataset/Flickr8k_Dataset/Flicker8k_Dataset'
    extract_features(dir_rel)
