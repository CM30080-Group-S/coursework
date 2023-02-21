import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from skimage.feature import match_template
from skimage.io import imread
from skimage.transform import pyramid_gaussian


def load_images(path):
    """
    Loads training images from a given path and returns them as a dictionary of numpy arrays, where the key is the filename.
    """
    if not path.endswith("/"):
        path += "/"
    try:
        images = {}
        for file in os.listdir(path):
            if file.endswith(".png"):
                images[file] = imread(path + file, as_gray=True)
        return images
    except FileNotFoundError:
        print(f"Error: The path {path} does not exist.")
        sys.exit(1)
    except NotADirectoryError:
        print(f"Error: The path {path} is not a directory.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: You do not have permission to access the path {path}.")
        sys.exit(1)
    except OSError:
        print(f"Error: The path {path} is not a valid path.")
        sys.exit(1)
    except:
        print(f"Error: An unexpected error occurred while loading the images.")
        sys.exit(1)

def gaussian_pyramid(image, levels):
    """
    Returns a gaussian pyramid for a given image and number of levels.
    """
    return list(pyramid_gaussian(image, max_layer=levels - 1))

def preprocess(image):
    """
    Preprocesses an image by setting the background to 0.
    """
    image[image < 0.5] = 0
    return image
    

def main():
    training_images = load_images("data/task2/training")
    
    # TODO 1: Extract variant creation from main function
    # Get gaussian pyramids for each image
    templates = defaultdict(list)
    for emoji, test_image in training_images.items():
        templates[emoji] = gaussian_pyramid(test_image, 4)    

    # Get rotated images for each level of each pyramid
    angles = np.linspace(30, 360, 11, endpoint=False)

    # Rotate each image in each pyramid
    # for emoji, emoji_templates in templates.items():
    #     rotated_templates = deepcopy(emoji_templates)
    #     for level, template in enumerate(emoji_templates):
    #         for angle in angles:
    #             print(f"Rotating {emoji} at level {level} by {angle} degrees...")
    #             rotated_templates.append(rotate(template, angle))
    #     templates[emoji] = rotated_templates
    #     print("Rotated templates:", len(templates[emoji]))
        
        

    
    # Load test images
    test_images = load_images("data/task2/test/images")


    # For each imag in the test set, find all matches above a threshold in the pyramids
    
        # Preprocess the test image
    print(test_images.keys())
    test_image = preprocess(test_images["test_image_1.png"])

    # For each pyramid
    matches = []
    for emoji, emoji_templates in templates.items():
        # For each rotated image
        for index, template in enumerate(emoji_templates):
            template = preprocess(template)
            print(f"Matching {emoji}, variant {index + 1}...")
            # Find matches
            result = match_template(test_image, template)
            
            if result.max() > 0.8:
                matches.append((emoji, result, index))
            

    # For each match, get the bounding box
    for emoji, result, index in matches:
        print(f"Found match for {emoji}, variant {index + 1} with score {result.max()}")
        # Get the bounding box
        ij = np.unravel_index(np.argmax(result), result.shape)
        x, y = ij[::-1]
        w, h = templates[emoji][index].shape
        print(f"Bounding box: ({x}, {y}), ({x + w}, {y + h})")

        # TODO 2: add in evaluation code. Load in the ground truth and compare the results to the ground truth.



if __name__ == "__main__":
    # TODO add command line parser. Update docs.
    parser = argparse.ArgumentParser()
    main()