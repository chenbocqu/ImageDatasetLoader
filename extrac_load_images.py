# Imports
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from skimage import transform

def extract_images(file_dir, classes, img_width, img_height, img_depth):
    """Function to extract data from images to float32 numpy arrays in [0, 255].
       If the dataset images havn't got the same shape it will be resized.
       This numpy arrays will have the shape NWHC.
       The data will be saved to npy files for later on, fast reading in.
       The dataset has to be stored one class per sub-folder in the given file_dir.
    Args:
        file_dir (str): Filepath to the dataset.
        classes (list of strs): List of class names.
        img_width, img_height, img_depth (int): Dimension of an image
    """
    # Allowed datatypes to read-in, just in case to skip files that are no images
    allowed_data_types = ["jpg", "jpeg", "png", "tiff", "bmp"]
    # Save the full-path for each classes sub-folder
    classes_dir = [file_dir + "/" + str(cl) + "/" for cl in classes]
    # Save all filenames for each class
    classes_files = [[class_dir + name for name in os.listdir(class_dir) 
        if name.split(".")[-1] in allowed_data_types] 
        for class_dir in classes_dir]
    # Number of images per class, to create empty numpy array
    num_samples_per_class = [len(class_files) for class_files in classes_files]

    # Create empty x and y array
    x = np.empty(shape=(np.sum(num_samples_per_class), img_width, img_height, img_depth), dtype=np.float32)
    y = np.empty(shape=(np.sum(num_samples_per_class)), dtype=np.float32)
    cnt = 0

    # For each class, iterate over all images.
    for class_num, (class_files) in enumerate(classes_files):
        for f in class_files:
            # If it will be a grayscale image
            if img_depth == 1:
                img = cv2.imread(f, 0)
            # If it will be a regular rgb image
            else:
                img = cv2.imread(f)
            if not img is None:
                # Resize image to defined image_size
                img = cv2.resize(img, (img_width, img_height))
                x[cnt] = img
                y[cnt] = class_num
                cnt += 1
            else:
                y[cnt] = -1

    # Delete corrupted images
    indx = [i for i in range(x.shape[0]) if y[i] == -1]
    x = np.delete(x, indx, axis=0)
    y = np.delete(y, indx, axis=0)

    # Store data to npy files
    np.save(file_dir + "/x.npy", x)
    np.save(file_dir + "/y.npy", y)

# Load images from npy file
def load_images(file_dir):
    """Function to load image data from npy files.
    Args:
        file_dir (str): Filepath to the dataset.
    Returns:
        x, y(ndarray): Image dataset
    """
    x = np.load(file_dir + "/x.npy")
    y = np.load(file_dir + "/y.npy")
    return x, y

# Example of usage
if __name__ == "__main__":
    file_dir = "PATH"
    img_size = 64
    img_depth = 3
    classes = ["cat", "dog"]

    extract_images(file_dir, classes, img_size, img_size, img_depth)
    x, y = load_images(file_dir)