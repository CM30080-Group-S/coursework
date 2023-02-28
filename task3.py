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
    

def main():
    # Load emojis
    emojis = load_images("data/task2/training/")
    print(f"Loaded {len(emojis)} emojis.")

    # Load test images
    test_images = load_images("data/task2/test/images")

    # Initialize the SIFT detector
    sift = SIFT(n_octaves=5, n_scales=4, )

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

            matches = match_descriptors(source_descriptors, target_descriptors, cross_check=True, max_ratio=0.7)
            print(f"Matches for image {test_image_name} and emoji {emoji_name}: {len(matches)}")
            # Output image showing lines between matching keypoints
            if len(matches) > 5:
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
    # TODO 1: Add command line arguments
    main()