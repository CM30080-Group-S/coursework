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
from skimage.transform import estimate_transform, matrix_transform
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
    Checks if the keypoints in the target image are local.
    """
    # Determine the centroid of the matches
    centroid = np.mean(keypoints, axis=0)
    # Calculate the distance of each match to the centroid
    distances = np.linalg.norm(keypoints - centroid, axis=1)
    # Check if the majority of the matches are within the threshold
    return np.sum(distances < locality_pixels) > confidence_threshold * len(distances)

def get_best_matches(matches, scene_keypoints, locality_pixels=64):
    """
    Returns the highest quality matches.

    Args:
        matches (np.ndarray): The indices of the matching keypoints between the target image and the template image.
        target_keypoints (np.ndarray): The keypoints of the test image.
        locality_pixels (int): The maximum distance of a match to the centroid of all matches.
    """
    # Get the best target keypoints
    keypoints = scene_keypoints[matches[:, 0]]

    # Determine the centroid of the matches
    centroid = np.mean(keypoints, axis=0)

    # Calculate the distance of each match to the centroid
    distances = np.linalg.norm(keypoints - centroid, axis=1)

    # Get the indices of the matches that are within the threshold
    indices = np.where(distances < locality_pixels)[0]
    return matches[indices]
    

def get_bounding_box(matches, template_keypoints, scene_keypoints, image_size):
    """
    Returns the bounding box of the template image.

    Args:
        matches (np.ndarray): The indices of the matching keypoints between the template image and the test image.
        template_keypoints (np.ndarray): The keypoints of the template image.
        target_keypoints (np.ndarray): The keypoints of the test image.
        image_size (tuple): The size of the test image.
    """

    # Find the best matches
    matches = get_best_matches(matches, scene_keypoints)

    src_points = template_keypoints[matches[:, 1]]
    dst_points = scene_keypoints[matches[:, 0]]

    # Create a trasformation matrix that maps the source image to the target image
    tform = estimate_transform("affine", src_points, dst_points)

    # Get the corners of the template image
    corners = np.array([[0,0], [image_size[0], image_size[1]]])

    # Transform the corners of the template image to the target image
    transformed_corners = tform(corners)

    # Swap the x and y coordinates
    transformed_corners = transformed_corners[:, [1, 0]]

    # Return the bounding box of the template image
    return transformed_corners

def output_bounding_boxes(bounding_boxes, scene_image, scene_image_name, output_path):
    """
    Saves the bounding box to a file.

    Args:
        bounding_boxes (dict(ndarray)): The bounding box of the template image.
        scene_image (np.ndarray): The test image.
        scene_image_name (str): The name of the test image.
        output_path (str): The path to the output directory.
    """
    # Create the output directory if it does not exist
    Path(output_path).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(scene_image, cmap=plt.cm.gray)
    for emoji_name, bounding_box in bounding_boxes.items():
        # Generate other corners for the bounding box
        bounding_box = np.array([
            bounding_box[0],
            [bounding_box[1][0], bounding_box[0][1]],
            bounding_box[1],
            [bounding_box[0][0], bounding_box[1][1]]
        ])

        # Pick a random color for the bounding box
        np.random.seed(sum([ord(c) for c in emoji_name]))
        color = np.random.rand(3,)

        # Draw a rectangular patch
        patch = plt.Polygon(bounding_box, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(patch)
        # Add the label to the patch
        ax.text(bounding_box[0][0] - 10, bounding_box[0][1], emoji_name, color=color)


    # Save the image
    plt.savefig(output_path + f"{scene_image_name}bounding_box.png")
    plt.close(fig)


def main(training_images_path,
    test_images_path,
    ground_truths_path,
    octaves,
    ratio,
    scales,
    threshold,
    show_boxes=False,
    show_matches=False,
    verbose=False):
    # Load emojis
    emojis = load_images(training_images_path)
    if verbose:
        print(f"Loaded {len(emojis)} emojis.")

    # Load test images
    scene_images = load_images(test_images_path)

    # Initialize the SIFT detector
    sift = SIFT(n_octaves=octaves, n_scales=scales)

    # For each test image, perform SIFT and match with emojis
    for scene_image_name, scene_image in tqdm(scene_images.items()):
        # if test_image_name != "test_image_10.png":
        #     continue
        sift.detect_and_extract(scene_image)
        scene_keypoints = sift.keypoints
        scene_descriptors = sift.descriptors
        bounding_boxes = {}

        for emoji_name, emoji in emojis.items():
            
            sift.detect_and_extract(emoji)
            template_keypoints = sift.keypoints
            template_descriptors = sift.descriptors

            matches = match_descriptors(scene_descriptors, template_descriptors, cross_check=True, max_ratio=0.7)

            if verbose:
                print(f"Matches for image {scene_image_name} and emoji {emoji_name}: {len(matches)}")
            
            if len(matches) > threshold:
                # Check locality of matches
                if are_local(scene_keypoints[matches[:, 0]]):
                    bounding_box = get_bounding_box(matches, template_keypoints, scene_keypoints, emoji.shape)
                    bounding_boxes[emoji_name] = bounding_box

            if show_matches:
                # Output image showing lines between matching keypoints
                fig, ax = plt.subplots(nrows=1, ncols=1)
                
                Path("output/task3/").mkdir(parents=True, exist_ok=True)
                plot_matches(image1=scene_image, image2=emoji, keypoints1=scene_keypoints, keypoints2=template_keypoints, matches=matches, only_matches=True, ax=ax)
                plt.savefig(f"output/task3/{scene_image_name}_{emoji_name}_matches.png")
                plt.close(fig)

        # TODO 3: Print the results
        
        output_bounding_boxes(bounding_boxes, scene_image, scene_image_name, f"output/task3/")

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