import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
from skimage.feature import match_template
from skimage.io import imread
from skimage.transform import pyramid_gaussian, rotate


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

def preprocess(image):
    """
    Preprocesses an image by setting the background to 0.
    """
    image[image < 0.5] = 0
    return image

# Memoise the function and store the results on disk
def generate_templates(image_class, image, levels, angles, verbose=False, use_cache=True):
    """
    Generates rotated templates for a given image, number of levels, and angles.
    """
    # Check if the templates have already been generated
    if use_cache:

        # Check if the cache directory exists
        if not os.path.exists(".cache"):
            os.mkdir(".cache")

        # Check how large the cache is in bytes
        cache_size_bytes = sum(os.path.getsize(f".cache/{f}") for f in os.listdir(".cache") if os.path.isfile(f".cache/{f}"))
        cache_size_mb = cache_size_bytes / 1000000
        if verbose:
            print(f"Cache size: {cache_size_mb} MB")
        
        if cache_size_mb > 1000 and verbose:
            print(f"Warning: The cache directory is getting large ({cache_size_mb} MB). Consider deleting some files.")

        # Generate hash using image_class, levels, and angles
        file_name = f"{image_class}_{levels}_{'_'.join([str(angle) for angle in angles])}"

        try:
            with open(f".cache/{file_name}.pkl", "rb") as f:
                if verbose:
                    print(f"Cache found for {image_class} with given parameters. Loading templates...")
                return pickle.load(f)
        except FileNotFoundError:
            if verbose:
                print("Cache not found. Generating templates...")
            pass

    # Generate gaussian pyramids
    pyramid = list(pyramid_gaussian(image, max_layer=levels - 1))

    # Rotate each image in each pyramid
    templates = []
    for level, template in enumerate(pyramid):
        for angle in angles:
            if verbose:
                print(f"Rotating at level {level} by {angle} degrees...")

            if angle == 0:
                templates.append(template)
                continue
            
            templates.append(rotate(template, angle))

    if use_cache:
        # If there already exist files with the same class, delete them
        if verbose:
            print(f"Clearing old templates from cache...")

        for file in os.listdir(".cache"):
            if file.startswith(image_class):
                os.remove(f".cache/{file}")

        # Save the templates to the cache
        if verbose:
            print(f"Saving templates to cache...")

        with open(f".cache/{file_name}.pkl", "wb") as f:
            pickle.dump(templates, f)

    return templates
    

def main():
    training_images = load_images("data/task2/training")
    
    # Get template variants of each emoji
    templates = defaultdict(list)
    for emoji, test_image in training_images.items():
        # TODO: add in command line arguments for levels and angles
        LEVELS = 4
        ANGLES = np.linspace(0, 360, 12, endpoint=False)
        templates[emoji] = generate_templates(emoji, test_image, LEVELS, [0], verbose=False)
    
    # Load test images
    test_images = load_images("data/task2/test/images")


    # Preprocess the test image
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
            print(result.max())
            
            if result.max() > 0.8:
                matches.append((emoji, result, index))
            
    # TODO: if there are no matches, and verbose is true, print "No matches found".

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