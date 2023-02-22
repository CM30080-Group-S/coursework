import argparse
import os
import pickle
import sys
from collections import defaultdict
from operator import itemgetter

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


def main(training_images_path, test_images_path, ground_truths_path, angles, levels, verbose=False):
    training_images = load_images(training_images_path)

    # Get template variants of each emoji
    templates = defaultdict(list)
    for emoji, training_image in training_images.items():
        roation_angles = np.linspace(0, 360, angles, endpoint=False)
        templates[emoji] = generate_templates(emoji, training_image, levels, roation_angles, verbose)

    # Load test images
    test_images = load_images(test_images_path)

    # Preprocess the test image
    test_image = preprocess(test_images["test_image_1.png"])

    # For each pyramid
    matches = []
    for emoji, emoji_templates in templates.items():
        # For each rotated image
        for index, template in enumerate(emoji_templates):
            template = preprocess(template)

            if verbose:
                print(f"Matching {emoji}, variant {index + 1}...")

            # Find matches
            result = match_template(test_image, template)

            if verbose:
                print(result.max())

            if result.max() > 0.8:
                matches.append((emoji, result, index))

    if len(matches) == 0 and verbose:
        print("No matches found")

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
    parser = argparse.ArgumentParser(
        description='''Task 2: Using intensity-based template matching to perform scale and rotation invariant identification of emojis and their bounding boxes. The accuracy is expressed as a percentage. The training images are located in the folder "data/task2/training", the test images are located in the folder "data/task2/test/images" and the corresponding annotations are located in the folder "data/task2/test/annotations"''',
    )
    parser.add_argument("training_images_path", help="Path to the training images")
    parser.add_argument("test_images_path", help="Path to the test images")
    parser.add_argument("ground_truths_path", help="Path to the ground truths")
    parser.add_argument("-a", "--angles", help="Angles between 0 and 360 to use for image rotations (default is 1)", default=1, type=int)
    parser.add_argument("-l", "--levels", help="Gaussian pyramid levels (default is 4)", default=4, type=int)
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    training_images_path, test_images_path, ground_truths_path, angles, levels, verbose = (
        itemgetter(
            "training_images_path",
            "test_images_path",
            "ground_truths_path",
            "angles",
            "levels",
            "verbose"
        )(vars(parser.parse_args()))
    )

    main(training_images_path, test_images_path, ground_truths_path, angles, levels, verbose)