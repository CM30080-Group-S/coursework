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

## Authors
- [Max Wood](https://maxwood.tech)
- [Rowan Bevan](https://github.com/RowanBevan)
- [Cameron Grant](https://github.com/cg-2611)

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements
- [SciKit Image](https://scikit-image.org/)