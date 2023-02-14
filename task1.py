from skimage import io
from skimage.transform import hough_line, hough_line_peaks
import os
import numpy as np
import matplotlib.pyplot as plt


def load_images(path):
    """
    Loads images from a given path and returns them as a list of numpy arrays.
    """
    if not path.endswith("/"):
        path += "/"

    images = []
    for file in os.listdir(path):
        if file.endswith(".png"):
            images.append(io.imread(path + file, as_gray=True))
    return images

def hough_transform(image):
    """
    Performs the hough transform on a given image.
    """
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(image, theta=tested_angles)
    return h, theta, d
    
def evaluate(predictions, ground_truth):
    """
    Evaluates the predictions of the model and returns the accuracy.
    """
    accuracies = []
    for i in range(len(predictions)):
        # Print the accuracy of the model
        error = predictions[i] - ground_truth[i]
        accuracy = error / ground_truth[i]
        accuracies.append(accuracy)
        print("Accuracy: " + str(accuracy))
    
    # Return the average accuracy
    return np.mean(accuracies)

def main(image_path):
    # For each image
    for image in load_images(image_path):
        # Perform hough transform on the image
        h, theta, d = hough_transform(image)
        
        # Plot the hough transform
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].set_title('Input image')

        lines = []
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            (x0, y0) = np.array([np.cos(angle), np.sin(angle)]) * dist
            lines.append((x0, y0, angle, dist))
        print(len(lines))
        


if __name__ == "__main__":
    main("task1/data/")