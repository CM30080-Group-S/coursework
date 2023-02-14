from skimage import io
from skimage.transform import hough_line, hough_line_peaks
from skimage.morphology import binary_erosion
import os
import numpy as np

def load_images(path):
    """
    Loads images from a given path and returns them as a dictionary of numpy arrays, where the key is the filename.
    """
    if not path.endswith("/"):
        path += "/"

    images = {}
    for file in os.listdir(path):
        if file.endswith(".png"):
            images[file] = io.imread(path + file, as_gray=True)
    return images

def load_ground_truth(path):
    angles = {}
    with open(path, "r") as f:
        for line in f:
            line = line.split(",")
            angles[line[0]] = float(line[1])
    return angles

def hough_transform(image):
    """
    Performs the hough transform on a given image.
    """
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(image, theta=tested_angles)
    return h, theta, d
    
def evaluate(predictions, ground_truth):
    """
    Evaluates the predictions of the model and returns the accuracy.
    """
    accuracies = []
    for i in range(len(predictions)):
        # Print the accuracy of the model
        absolute_error = np.abs(predictions[i] - ground_truth[i])
        relative_error = absolute_error / ground_truth[i]
        accuracies.append(relative_error)

        # Express the accuracy as a percentage
        relative_error *= 100
        print(f"Accuracy for image {i + 1}: {100 - relative_error:.2f}%")
    
    # Return the average accuracy
    return np.mean(accuracies)

def preprocess(image):
    """
    Preprocesses the image by removing the background.
    """

    # Binarize the image
    image = image > 0.25

    # Binary erosion
    image = binary_erosion(image)

    return image

def get_angle(line_1, line_2):
    """
    Returns the angle between two lines.
    """
    x1, y1, angle_1, dist_1 = line_1
    x2, y2, angle_2, dist_2 = line_2
    return np.abs(angle_1 - angle_2)


def main(image_path):
    ground_truth = load_ground_truth("task1/data/list.txt")
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

        print(f"Found {len(lines)} lines for image {image_name}")

        angle = np.rad2deg(get_angle(lines[0], lines[1]))
        predictions[image_name] = angle
    
    # Sort the predictions and ground truth by filename
    # Print the predictions and ground truth for each image
    for image_name in sorted(predictions.keys()):
        print(f"Prediction for image {image_name}: {predictions[image_name]}")
        print(f"Ground truth for image {image_name}: {ground_truth[image_name]}")

    # Evaluate the predictions
    evaluate(predictions, ground_truth)
    

if __name__ == "__main__":
    main("task1/data/")