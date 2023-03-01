import argparse
import os
import pickle
import sys
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import SIFT, match_descriptors, plot_matches
from skimage.filters import gaussian
from skimage.io import imread
from tqdm import tqdm


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

def are_local(keypoints, locality_pixels=64, confidence_threshold=0.6):
    """
    Checks if the matches are local.
    """
    # Determine the centroid of the matches
    centroid = np.mean(keypoints, axis=0)
    # Calculate the distance of each match to the centroid
    distances = np.linalg.norm(keypoints - centroid, axis=1)
    print(distances)
    # Check if the majority of the matches are within the threshold
    return np.sum(distances < locality_pixels) > confidence_threshold * len(distances)


def main(
    training_images_path,
    test_images_path,
    ground_truths_path,
    octaves,
    ratio,
    scales,
    threshold,
    show_boxes=False,
    show_matches=False,
    verbose=False
):
    # Load emojis
    emojis = load_images(training_images_path)
    if verbose:
        print(f"Loaded {len(emojis)} emojis.")

    # Load test images
    test_images = load_images(test_images_path)

    # Initialize the SIFT detector
    sift = SIFT(n_octaves=octaves, n_scales=scales)

    # For each test image, perform SIFT and match with emojis
    for test_image_name, test_image in tqdm(test_images.items()):
        # if test_image_name != "test_image_10.png":
        #     continue
        sift.detect_and_extract(test_image)
        source_keypoints = sift.keypoints
        source_descriptors = sift.descriptors
        found=[]

        for emoji_name, emoji in emojis.items():
            sift.detect_and_extract(emoji)
            target_keypoints = sift.keypoints
            target_descriptors = sift.descriptors

            matches = match_descriptors(source_descriptors, target_descriptors, cross_check=True, max_ratio=ratio)

            if verbose:
                print(f"Matches for image {test_image_name} and emoji {emoji_name}: {len(matches)}")

            if show_matches:
                # Output image showing lines between matching keypoints
                if len(matches) > threshold:
                    # Check locality of matches
                    if are_local(source_keypoints[matches[:, 0]]):
                        fig, ax = plt.subplots(nrows=1, ncols=1)
                        found.append(emoji_name)
                        # Write the image to disk
                        Path("data/task3/").mkdir(parents=True, exist_ok=True)
                        plot_matches(image1=test_image, image2=emoji, keypoints1=source_keypoints, keypoints2=target_keypoints, matches=matches, only_matches=True, ax=ax)
                        plt.savefig(f"data/task3/{test_image_name}_{emoji_name}.png")
                        plt.close(fig)

                    # TODO 2: Get bounding box

        # TODO 3: Print the results

        print(test_image_name)
        for i in found:
            print(i)

        # For each emoji, perform SIFT and match with the test image
    # For each match found, draw a rectangle around the emoji

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Task 3: Using SIFT feature matching to perform identification of emojis and their bounding boxes. The accuracy is expressed as a percentage. The training images are located in the folder "data/task2/training", the test images are located in the folder "data/task3/test/images" and the corresponding annotations are located in the folder "data/task3/test/annotations"''',
    )
    parser.add_argument("training_images_path", help="Path to the training images")
    parser.add_argument("test_images_path", help="Path to the test images")
    parser.add_argument("ground_truths_path", help="Path to the ground truths")
    parser.add_argument("-b", "--boxes", help="Show the bounding boxes", action="store_true")
    parser.add_argument("-m", "--matches", help="Show the feature matches", action="store_true")
    parser.add_argument("-o", "--octaves", help="The number of SIFT octaves (default is 5)", default=5, type=int)
    parser.add_argument("-r", "--ratio", help="The maximum ratio distances used when matching (default is 0.7)", default=0.7, type=float)
    parser.add_argument("-s", "--scales", help="The number of scales per SIFT octave (default is 4)", default=4, type=int)
    parser.add_argument("-t", "--threshold", help="The number of feature matches required for a class prediction (default is 5)", default=5, type=int)
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    training_images_path, test_images_path, ground_truths_path, octaves, ratio, scales, threshold, show_boxes, show_matches, verbose = (
        itemgetter(
            "training_images_path",
            "test_images_path",
            "ground_truths_path",
            "boxes",
            "matches",
            "octaves",
            "ratio",
            "scales",
            "threshold",
            "verbose"
        )(vars(parser.parse_args()))
    )

    main(
        training_images_path,
        test_images_path,
        ground_truths_path,
         octaves,
        ratio,
        scales,
        threshold,
        show_boxes,
        show_matches,
        verbose
    )