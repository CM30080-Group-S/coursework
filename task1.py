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

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].set_title('Input image')
        ax[0].set_axis_off()
        
        angle_step = 0.5 * np.diff(theta).mean()
        dist_step = 0.5 * np.diff(d).mean()
        bounds = [
            np.rad2deg(theta[0] - angle_step),
            np.rad2deg(theta[-1] + angle_step),
            d[-1] + dist_step,
            d[0] - dist_step,
        ]
        ax[1].imshow(np.log(1 + h), extent=bounds, cmap='gray', aspect='auto')
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(image, cmap=plt.cm.gray)
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')


        lines = []
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            (x0, y0) = np.array([np.cos(angle), np.sin(angle)]) * dist
            lines.append((x0, y0, angle, dist))
            ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2), color='r')
        print(len(lines))
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main("task1/data/")