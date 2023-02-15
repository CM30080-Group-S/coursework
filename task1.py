from skimage import io
from skimage.transform import hough_line, hough_line_peaks
from skimage.morphology import binary_erosion
import os
import numpy as np
import sys
import argparse
from operator import itemgetter

def load_images(path):
    """
    Loads images from a given path and returns them as a dictionary of numpy arrays, where the key is the filename.
    """
    if not path.endswith("/"):
        path += "/"

    try:
        images = {}
        for file in os.listdir(path):
            if file.endswith(".png"):
                images[file] = io.imread(path + file, as_gray=True)
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

def load_ground_truth(path):
    try:
        angles = {}
        with open(path, "r") as f:
            for line in f:
                line = line.split(",")
                angles[line[0]] = float(line[1])
        return angles
    except FileNotFoundError:
        print(f"Error: The path {path} does not exist.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: You do not have permission to access the path {path}.")
        sys.exit(1)
    except OSError:
        print(f"Error: The path {path} is not a valid path.")
        sys.exit(1)
    except:
        print(f"Error: An unexpected error occurred while loading the ground truth.")
        sys.exit(1)

def hough_transform(image):
    """
    Performs the hough transform on a given image.
    """
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(image, theta=tested_angles)
    return h, theta, d
    
def evaluate(predictions, ground_truth, verbose=False):
    """
    Evaluates the predictions of the model and returns the accuracy.
    """
    accuracies = []
    for i in range(len(predictions)):
        # Print the accuracy of the model
        absolute_error = np.abs(predictions[i] - ground_truth[i])
        relative_error = absolute_error / ground_truth[i]
        accuracy = 1 - relative_error
        accuracies.append(accuracy)

        if verbose:
            # Express the accuracy as a percentage
            print(f"Accuracy for image {i + 1}: {accuracy * 100:.2f}%")
    
    # Return the average accuracy
    return np.mean(accuracies)

def preprocess(image):
    """
    Preprocesses the image.
    """

    # Binarize the image
    image = image > 0.25

    # Binary erosion
    image = binary_erosion(image)

    return image

def get_angle(line_1, line_2):
    """
    Returns the angle between two lines in radians.
    """
    angle_1 = line_1[2]
    angle_2 = line_2[2]
    return np.abs(angle_1 - angle_2)


def main(image_path, ground_truth_path, verbose=False):
    ground_truth = load_ground_truth(ground_truth_path)
    predictions = {}
    # For each image
    for image_name, image in load_images(image_path).items():
        # Preprocess the image
        image = preprocess(image)

        # Perform hough transform on the image
        h, theta, d = hough_transform(image)

        lines = []
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            (x0, y0) = np.array([np.cos(angle), np.sin(angle)]) * dist
            lines.append((x0, y0, angle, dist))

        if verbose:
            print(f"Found {len(lines)} lines for image {image_name}")

        assert len(lines) == 2, "Expected two lines"
        # Get the angle between the first two lines
        angle = np.rad2deg(get_angle(lines[0], lines[1]))
        predictions[image_name] = angle

    # Sort the predictions and ground truth by filename
    predictions = [predictions[key] for key in sorted(predictions.keys())]
    ground_truth = [ground_truth[key] for key in sorted(ground_truth.keys())]

    # Evaluate the predictions
    average_accuracy = evaluate(predictions, ground_truth, verbose)
    if verbose:
        print(f"Average accuracy: {average_accuracy * 100:.2f}%")
    return average_accuracy
    

if __name__ == "__main__":
    # TODO get this to work with single images and ground truth values
    parser = argparse.ArgumentParser(
        description='''Task 1: Measuring the angle between two lines in an image using the Hough transform and the RANSAC algorithm. The angle is measured in degrees and the accuracy is measured as the relative error between the predicted angle and the ground truth angle. The accuracy is expressed as a percentage. The average accuracy is the average of the accuracies of all images. The images are located in the folder "data" and the ground truth is located in the file "list.txt''',
    )
    parser.add_argument("image_path", help="Path to the images")
    parser.add_argument("ground_truth_path", help="Path to the ground truth")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    image_path, ground_truth_path, verbose = itemgetter("image_path", "ground_truth_path", "verbose")(vars(parser.parse_args()))
    main(image_path, ground_truth_path, verbose)