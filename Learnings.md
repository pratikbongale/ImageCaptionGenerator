### Things I learned while developing this project

------------
If you want git ignore to ignore some file that already exists:
1. Commit all pending changes, then run this command:
```
git rm -r --cached .
```
2. This removes everything from the index, then just run:
```
git add .
```
3. Commit it:
```
git commit -m ".gitignore is now working"
```
-----------
Keras downloads its models into directory: 
~/.keras/models/ 
all models are stored here the first time they are downloaded.
I had to run the download process in background, because foreground process failed
everytime because my network resets the connection if we are downloading 
from the same source for too long. I had to run it in background, so I created a simple scrips 
and ran it in background.
```
$ nohup python dl_vgg16.py &
```
This will download the model into `~/.keras/models/` and store logs in nohup.out file.

------------------

```python
pil_img = load_img(fpath, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))   # load an img in Python Imaging Library(PIL) format
img_arr = img_to_array(pil_img)     # 3D numpy array (x, y, layer(RGB))
img_shape = img_arr.shape
flat_img_arr = img_arr.reshape((1, img_shape[0], img_shape[1], img_shape[2]))
image = preprocess_input(flat_img_arr)  # preprocessing powered for imagenet
features = model.predict(image, verbose=0)
```
**load_img** loads given image into PIL format which is a common image format to which multiple image formats(.jpg, .gif, .bmp) can be converted to just like java byte code

**img_to_arr** converts it from PIL image instance to nparray

**reshape((1, height, width, channels))** shape of one image is `(height, width, channels)`, if we want to process a batch of images we use additional dimension `(samples, size1,size2,channels)` 

**preprocess_input** Some models use images with values ranging from 0 to 1. Others from -1 to +1. Others use the "caffe" style, that is not normalized, but is centered.

-----------------
string.join(iterable)
returns a string concatenated with the elements of an iterable separated by arg string used to make the call.

example:
```python
numList = ['1', '2', '3', '4']
seperator = ', '
print(seperator.join(numList))
```
------------------
have a default value if something doesnt exists in a dictionary
```python
from collections import defaultdict
arr = ['a','b','c','d']

d = defaultdict()
for i, ele in enumerate(arr):
    d[i] = ele

```
-------------------
get rid of punctuations in a string
```python
import string

text = "Hi! I am mack, I am awesome;"
# make a translation table using the string datatype "str"
tab = str.maketrans('', '', string.punctuation)
desc = [w.translate(tab) for w in text]
' '.join(desc)  # join desc using spaces
```            

The first and second arguments should actually give the mappings
whatever is in the third argument is mapped to None

```python
# first string
firstString = "abc"
secondString = "ghi"
thirdString = "ab"

string = "abcdef"
translation = string.maketrans(firstString, secondString, thirdString)

# translate string
print("Translated string:", string.translate(translation))  # prints idef
```
--------------------
using a tokenizer - unknown words are skipped
```python
from keras.preprocessing.text import Tokenizer
tok = Tokenizer()
tok.fit_on_texts(["this comment is not toxic"]) 
print(tok.texts_to_sequences(["this comment is not toxic"])) 
print(tok.texts_to_sequences(["this very long comment is not toxic"]))
```
output:
```
Using TensorFlow backend.
[[1, 2, 3, 4, 5]]
[[1, 2, 3, 4, 5]]
```
---------------------------
Difference between `__init__` and `__call__`

```python
class Foo:
    def __init__(self, a, b, c):
        # initialize a newly created object
        pass

x = Foo(1, 2, 3) # __init__

class Foo:
    def __call__(self, a, b, c):
        # function call "operator"
        pass

x = Foo()
x(1, 2, 3) # __call__
```


