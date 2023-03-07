import argparse
import os
import pickle
import sys
from time import time
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

import numpy as np
from skimage.feature import match_template
from skimage.io import imread
from skimage.transform import pyramid_gaussian, rotate
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

def preprocess(image):
    """
    Preprocesses an image by setting the background to 0.
    """
    image[image < 0.5] = 0
    return image

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


def compute_bounding_box(match,templates):
    """
    Returns the bottom left and top right coordinates of the coordinates.
    """
    emoji, result, index = match
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    w, h = templates[emoji][index].shape
    return ((x,y),(x+w,y+h))


def compute_iou(box_1,box_2):
    """
    Compute the intersection over union of the matching bounding boxes.
    """

    # bl is the bottom left of the given box and tr is the top right
    bl, tr, x, y = (0, 1, 0, 1)

    box_1_area = (box_1[tr][x] - box_1[bl][x]) * (box_1[tr][y] - box_1[bl][y])
    box_2_area = (box_2[tr][x] - box_2[bl][x]) * (box_2[tr][y] - box_2[bl][y])

    intersect = (min(box_1[tr][x], box_2[tr][x]) - max(box_1[bl][x], box_2[bl][x])) * (min(box_1[tr][y], box_2[tr][y]) - max(box_1[bl][y], box_2[bl][y]))

    # If there is no intersection
    if intersect < 0:
        intersect = 0
    union = box_1_area + box_2_area - intersect
    assert union > 0, "Union must be greater than zero"
    return intersect / union


def evaluate(matches, templates, annotations_file):
    # Read in the data from the annotations
    annotations = {}
    with open(annotations_file, "r") as f:
        lines = f.readlines()
        for annotation in lines:
            annotation = annotation.split(",", 1)
            class_name = annotation[0]
            bounding_box = annotation[1].strip()

            bounding_box = bounding_box.split(",")

            start_x = int(bounding_box[0].strip()[1:])
            start_y = int(bounding_box[1].strip()[:-1])
            end_x = int(bounding_box[2].strip()[1:])
            end_y = int(bounding_box[3].strip()[:-1])

            annotations[class_name] = ((start_x, start_y), (end_x, end_y))

    true_positives = 0
    false_positives = 0
    accuracy = []
    # Determine if we have a: true positive, false positive for each match
    for match in matches:
        emoji = match[0]
        bounding_box = compute_bounding_box(match, templates)

        # Extract the emoji name from the emoji file name
        emoji = emoji.rsplit(".")[0].split("-",1)[1]

        # Determine if we have a: true positive, false positive
        if emoji in annotations.keys():
            true_positives += 1
            accuracy.append(compute_iou(bounding_box, annotations[emoji]))
        else:
            false_positives += 1
            accuracy.append(0)

    # Loop through the annotatioins that were not matched
    for annotation in annotations.keys():
        if annotation not in [(match[0].rsplit(".")[0].split("-",1)[1]) for match in matches]:
            accuracy.append(0)
    
    # Compute the intersection over union of the bounding boxes
    overall_accuracy = sum(accuracy) / len(accuracy)

    return true_positives, false_positives, overall_accuracy


def main(training_images_path, test_images_path, ground_truths_path, angles, levels, threshold, verbose=False):
    training_images = load_images(training_images_path)

    # Get template variants of each emoji
    templates = defaultdict(list)
    for emoji, training_image in training_images.items():
        roation_angles = np.linspace(0, 360, angles, endpoint=False)
        templates[emoji] = generate_templates(emoji, training_image, levels, roation_angles, verbose)

    # Load test images
    test_images = load_images(test_images_path)
    total_tp = 0
    total_fp = 0
    total_acc = []
    times_taken = []
    for key, test_image in tqdm(test_images.items()):
        start = time()
        # Preprocess the test image
        test_image = preprocess(test_image)
        annotation_file_path = ground_truths_path + '/' + (key.split('.')[0])
        if (Path(annotation_file_path + '.txt')).exists():
            annotation_file_path += '.txt'
        elif (Path(annotation_file_path + '.csv')).exists():
            annotation_file_path += '.csv'
        else:
            raise FileNotFoundError(f'No matching annotations found for test image {key}, continuing to next test image...')

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

                if result.max() > threshold:
                    matches.append((emoji, result, index))

        if len(matches) == 0 and verbose:
            print("No matches found")

        # For each match, get the bounding box
        for match in matches:
            emoji, result, index = match
            if verbose:
                print(f"Found match for {emoji}, variant {index + 1} with score {result.max()}")
            # Get the bounding box
            bounding_box = compute_bounding_box(match,templates)
            # TODO 2: Bounding boxes
            if verbose:
                print(f"Bounding box: {bounding_box}")
        end = time()
        times_taken.append(end - start)
        tp, fp, acc = evaluate(matches, templates, annotation_file_path)
        total_tp += tp
        total_fp += fp
        total_acc.append(acc)

    print(f"True positive rate: {total_tp / (total_tp + total_fp)} (total: {total_tp})")
    print(f"False positive rate: {total_fp / (total_tp + total_fp)} (total: {total_fp})")
    print(f"Accuracy: {np.mean(total_acc)}")
    print(f"Average time taken: {np.mean(times_taken):.2f}s")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''Task 2: Using intensity-based template matching to perform scale and rotation invariant identification of emojis and their bounding boxes. The accuracy is expressed as a percentage. The training images are located in the folder "data/task2/training", the test images are located in the folder "data/task2/test/images" and the corresponding annotations are located in the folder "data/task2/test/annotations"''',
    )
    parser.add_argument("training_images_path", help="Path to the training images")
    parser.add_argument("test_images_path", help="Path to the test images")
    parser.add_argument("ground_truths_path", help="Path to the ground truths")
    parser.add_argument("-a", "--angles", help="Angles between 0 and 360 to use for image rotations (default is 1)", default=1, type=int)
    parser.add_argument("-l", "--levels", help="Gaussian pyramid levels (default is 4)", default=4, type=int)
    parser.add_argument("-t", "--threshold", help="Threshold required for a match (default is 0.95)", default=0.95, type=float)
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    training_images_path, test_images_path, ground_truths_path, angles, levels, threshold, verbose = (
        itemgetter(
            "training_images_path",
            "test_images_path",
            "ground_truths_path",
            "angles",
            "levels",
            "threshold",
            "verbose"
        )(vars(parser.parse_args()))
    )

    main(training_images_path, test_images_path, ground_truths_path, angles, levels, threshold, verbose)
