Error line:
`features = model.predict(image, verbose=0)`
ValueError: Error when checking : expected input_1 to have shape (224, 224, 3) but got array with shape (500, 333, 3)

Solution:
Force the loading of the photo to have the same pixel dimensions as the VGG model, which are 224 x 224 pixels.
load_img(fpath, target_size=(224, 224))     # set the target size = (img_height, img_width)

Referrences:
https://machinelearningmastery.com/prepare-photo-caption-dataset-training-deep-learning-model/
https://keras.io/applications/#vgg16 - examples section


