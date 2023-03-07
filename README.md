# Computer Vision Coursework - CM30080

Solving computer vision problems using Python.

## Pre-requisites
- [Python 3.10](https://www.python.org/downloads/)

## Installation
- Clone the repository
```
https://github.com/CM30080-Group-S/coursework.git
```
- Install the required packages
```
pip install -r requirements.txt
```

## Task 1
Given an image with a pair of lines, find the smaller angle between the lines. We binarise the image and perform an erosion pass to reduce the width of the lines. We then use the Hough transform to find the lines in the image. We then find the angle between the lines using the slope of the lines.

### Usage
```
python task1.py [-v] <images_path> <ground_truth_path>
```
- `-v` - Optional flag to enable verbose mode
- `<image_path>` - Path to the image
- `<ground_truth_path>` - Path to the ground truth file (CSV)

### Example
```
python task1.py data/task1/ data/task1/ground_truth.txt
```

## Task 2
Given an image containing rotated and scaled variants of images from a given bank, use intensity-based template matching to identify the variants in the original image. We preprocess the images by setting the background to 0. We then take the bank of images and create a [Gaussian pyramid](https://en.wikipedia.org/wiki/Pyramid_(image_processing)) for each image. We then take each scaled image from the pyramid and generate multiple rotations at varying angles. Finally, we take these templates and "slide" them over the test images and use intensity-based matching to identify the scaled and rotated images and their bounding boxes.

### Usage
```
python task2.py [-a] [-l] [-t] [-v] <training_images> <test_image_path> <ground_truth_path>
```
- `-a` - Optional flag to specify the number of angles between 0 and 360 to use for image rotations, the default is 1
- `-l` - Optional flag to specify the number of levels of the Gaussian pyramid, the default is 4
- `-t` - Optional flag to specify the threshold required for a match
- `-v` - Optional flag to enable verbose mode
- `<training_images_path>` - Path to the training images
- `<test_images_path>` - Path to the test images
- `<ground_truths_path>` - Path to the ground truth files (CSV)

### Example
```
python task2.py data/task2/training data/task2/test/images data/task2/test/annotations
```

## Task 3
Given an image containing rotated and scaled variants of images from a given bank, use [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) feature matching to identify the variants in the original image. We detect the keypoints in an image and extract their descriptors. For each test image, we iterate over the emoji images, and perform the same keypoint detection on the emoji. We then compare and match the descriptors of the keypoints of the emoji image and the test image. Any class that has an amount of matches greater than a given threshold are then predicted for the test image.

### Usage
```
python task3.py [-b] [-m] [-o] [-r] [-s] [-t] [-v] <training_images> <test_image_path> <ground_truth_path>
```
- `-b` - Optional flag to show the bounding boxes
- `-m` - Optional flag to show the feature matches
- `-o` - Optional flag to specify the number of SIFT octave, the default is 5
- `-r` - Optional flag to specify the maximum ratio distances used when matching, the default is 0.7
- `-s` - Optional flag to specify the number of scales per SIFT octave, the default is 4
- `-t` - Optional flag to specify the number of feature matches required for a class prediction, the default is 5
- `-v` - Optional flag to enable verbose mode
- `<training_images_path>` - Path to the training images
- `<test_images_path>` - Path to the test images
- `<ground_truths_path>` - Path to the ground truth files (CSV)

### Example
```
python task3.py data/task2/training data/task3/test/images data/task3/test/annotations
```

## Authors
- [Max Wood](https://maxwood.tech)
- [Rowan Bevan](https://github.com/RowanBevan)
- [Cameron Grant](https://github.com/cg-2611)

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements
- [SciKit Image](https://scikit-image.org/)