import argparse
import os
import sys
from operator import itemgetter
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import SIFT, match_descriptors, plot_matches
from skimage.io import imread
from skimage.transform import estimate_transform
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
    tform = estimate_transform("similarity", src_points, dst_points)

    # Get the corners of the template image
    corners = np.array([[0,0], [0, image_size[0]], [image_size[1], image_size[0]], [image_size[1], 0]])

    # Transform the corners of the template image to the target image
    transformed_corners = tform(corners)

    # Swap the x and y coordinates
    transformed_corners = transformed_corners[:, [1, 0]]

    bl = (np.min(transformed_corners[:, 0]), np.max(transformed_corners[:, 1]))
    tr = (np.max(transformed_corners[:, 0]), np.min(transformed_corners[:, 1]))

    tl = (np.min(transformed_corners[:, 0]), np.min(transformed_corners[:, 1]))
    br = (np.max(transformed_corners[:, 0]), np.max(transformed_corners[:, 1]))

    print (bl, tr)
    # Return the bounding box of the template image
    return (bl, tl, tr, br)

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
        # Pick a random color for the bounding box
        np.random.seed(sum([ord(c) for c in emoji_name]))
        color = np.random.rand(3,)

        # Draw a rectangular patch
        patch = plt.Polygon(bounding_box, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(patch)
        # Add the label to the patch
        ax.text(bounding_box[0][0], bounding_box[0][1] + 10, emoji_name, color=color)

    # Save the image
    plt.savefig(output_path + f"{scene_image_name}_bounding_box.png")
    plt.close(fig)

def parse_emoji_name(emoji_name):
    return emoji_name.rsplit(".")[0].split("-",1)[1]

def compute_iou(box_1,box_2):
    """
    Compute the intersection over union of the matching bounding boxes.
    """

    # tl is the top left of the given box and br is the bottom right
    tl, br, x, y = (0, 1, 0, 1)

    box_1_area = (box_1[br][x] - box_1[tl][x]) * (box_1[br][y] - box_1[tl][y])
    box_2_area = (box_2[br][x] - box_2[tl][x]) * (box_2[br][y] - box_2[tl][y])

    intersect = (min(box_1[br][x], box_2[br][x]) - max(box_1[tl][x], box_2[tl][x])) * (min(box_1[br][y], box_2[br][y]) - max(box_1[tl][y], box_2[tl][y]))

    # If there is no intersection
    if intersect < 0:
        intersect = 0
    union = box_1_area + box_2_area - intersect
    assert union > 0, "Union must be greater than zero"
    return intersect / union

def evaluate(bounding_boxes, annotations_file):
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
    for class_name, bounding_box in bounding_boxes.items():
        # Extract the emoji name from the emoji file name
        emoji = class_name
        # Determine if we have a: true positive, false positive
        if emoji in annotations.keys():
            true_positives += 1
            print("emoji:",emoji)
            accuracy.append(compute_iou((bounding_box[1], bounding_box[3]), annotations[emoji]))
        else:
            false_positives += 1
            accuracy.append(0)

    # Loop through the annotatioins that were not matched
    for annotation in annotations.keys():
        if annotation not in bounding_boxes.keys():
            accuracy.append(0)

    print("accuracy:", accuracy)
    # Compute the intersection over union of the bounding boxes
    overall_accuracy = sum(accuracy) / len(accuracy)

    return true_positives, false_positives, overall_accuracy



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

    times_taken = []
    total_tp = 0
    total_fp = 0
    total_acc = []

    # For each test image, perform SIFT and match with emojis
    for scene_image_name, scene_image in tqdm(scene_images.items()):

        annotation_file_path = ground_truths_path + '/' + (scene_image_name.split('.')[0])
        if (Path(annotation_file_path + '.txt')).exists():
            annotation_file_path += '.txt'
        elif (Path(annotation_file_path + '.csv')).exists():
            annotation_file_path += '.csv'
        else:
            raise FileNotFoundError(f'No matching annotations found for test image {scene_image_name}, continuing to next test image...')

        start = time()
        # if test_image_name != "test_image_10.png":
        #     continue
        sift.detect_and_extract(scene_image)
        scene_keypoints = sift.keypoints
        scene_descriptors = sift.descriptors
        bounding_boxes = {}

        for emoji_name, emoji in emojis.items():

            emoji_name = parse_emoji_name(emoji_name)
            
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
        end = time()
        times_taken.append(end - start)
        print("Bounding boxes: ", bounding_boxes)
        tp, fp, acc = evaluate(bounding_boxes, annotation_file_path)
        total_tp += tp
        total_fp += fp
        total_acc.append(acc)
    
        output_bounding_boxes(bounding_boxes, scene_image, scene_image_name, f"output/task3/")
    
    print(total_acc)
    print(f"True positive rate: {total_tp / (total_tp + total_fp)} (total: {total_tp})")
    print(f"False positive rate: {total_fp / (total_tp + total_fp)} (total: {total_fp})")
    print(f"Accuracy: {np.mean(total_acc)}")
    print(f"Average time taken: {np.mean(times_taken):.2f}s")
    print(f"Average time taken: {round(np.mean(times_taken), 2)}s")

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
    training_images_path, test_images_path, ground_truths_path, show_boxes, show_matches, octaves, ratio, scales, threshold, verbose = (
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
