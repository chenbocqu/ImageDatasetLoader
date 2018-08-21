# Imports
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from skimage import transform

def extract_images(file_dir, classes, img_width, img_height, img_depth):
    """Function to extract data from images to numpy array.
       If the dataset images havn't got the same shape it will be resized.
       This numpy array will have the shape (num_samples, width, height, channels).
       The numpy array data will be stored to npy files for later on, fast read-in data.
       The dataset has to be stored one class per sub-folder in the given file_dir.
       The values are in [0, 255] ant the targets are the class nums.
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
    classes_files = [[name for name in os.listdir(class_dir) 
        if name.split(".")[-1] in allowed_data_types] 
        for class_dir in classes_dir]
    # Number of images per class, to create empty numpy array
    num_samples_per_class = [len(class_files) for class_files in classes_files]

    # Create empty x and y array
    x = np.empty(shape=(np.sum(num_samples_per_class), img_width, img_height, img_depth), dtype=np.float32)
    y = np.zeros(shape=(np.sum(num_samples_per_class)), dtype=np.float32)
    cnt = 0

    # For each class, iterate over all images.
    for class_num, (class_files, class_dir) in enumerate(zip(classes_files, classes_dir)):
        for f in class_files:
            # Load all images with cv2 function as uint8 type
            try:
                # If it will be a grayscale image
                if img_depth == 1:
                    img = cv2.imread(class_dir + f, 0)
                # If it will be a regular rgb image
                else:
                    img = cv2.imread(class_dir + f)
                # Resize image to defined image_size
                img = transform.resize(img, (img_width, img_height, img_depth))
                x[cnt] = img
                y[cnt] = class_num
            # Exception will be thrown if image is corrupted
            except:
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
    file_dir = "C:/Users/Jan/Documents/DogsAndCats"
    img_size = 64
    img_depth = 3
    classes = ["cat", "dog"]
    extract_images(file_dir, classes, img_size, img_size, img_depth)
    x, y = load_images(file_dir)

    indx = np.random.randint(0, x.shape[0], 10)
    imgs = x[indx]
    labels = y[indx]

    for img, label in zip(imgs, labels):
        print(img)
        if img_depth == 1:
            plt.imshow(img.reshape((img_size, img_size)))
        else:
            plt.imshow(img)
        plt.title(label)
        plt.show()