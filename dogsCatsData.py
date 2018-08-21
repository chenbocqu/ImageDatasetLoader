# Imports
import matplotlib.pyplot as plt
import os
import cv2
from skimage import transform
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
 
file_dir = "C:/Users/Jan/Documents/DogsAndCats"
img_size = 64
img_depth = 3

# Load Cats (0) and Dogs (1) from images to NumpyArray
def extract_cats_vs_dogs(file_dir):
    cats_dir = file_dir + "/cat/"
    dogs_dir = file_dir + "/dog/"

    print("Delete no jpg images!")
    for f in os.listdir(cats_dir):
        if f.split(".")[-1] != "jpg":
            print("Removing file: ", f)
            os.remove(cats_dir + f)

    print("Delete no jpg images!")
    for f in os.listdir(dogs_dir):
        if f.split(".")[-1] != "jpg":
            print("Removing file: ", f)
            os.remove(dogs_dir + f)

    num_cats = len([name for name in os.listdir(cats_dir)])
    num_dogs = len([name for name in os.listdir(dogs_dir)])

    x = np.empty(shape=(num_cats + num_dogs, img_size, img_size, img_depth), dtype=np.float32)
    y = np.zeros(shape=(num_cats + num_dogs), dtype=np.float32)
    cnt = 0

    print("Start reading in cat images!")
    for f in os.listdir(cats_dir):
        try:
            if img_depth == 1:
                img = cv2.imread(cats_dir + f, 0)
            else:
                img = cv2.imread(cats_dir + f)
            x[cnt] = transform.resize(img, (img_size, img_size, img_depth))
            y[cnt] = 0
            cnt += 1
        except:
            pass

    print("Start reading in dog images!")
    for f in os.listdir(dogs_dir):
        try:
            if img_depth == 1:
                img = cv2.imread(dogs_dir + f, 0)
            else:
                img = cv2.imread(dogs_dir + f)
            x[cnt] = transform.resize(img, (img_size, img_size, img_depth))
            y[cnt] = 1
            cnt += 1
        except:
            pass

    np.save(file_dir + "/x.npy", x)
    np.save(file_dir + "/y.npy", y)

def load_cats_vs_dogs(file_dir):
    x = np.load(file_dir + "/x.npy")
    y = np.load(file_dir + "/y.npy")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return (x_train, y_train), (x_test, y_test)

class CATSDOGS:
    x_train, y_train, x_test, y_test = None, None, None, None
    train_size, test_size = 0, 0

    def __init__(self, file_dir):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_cats_vs_dogs(file_dir)
        # reshape
        self.x_train = self.x_train.reshape(self.x_train.shape[0], img_size, img_size, img_depth)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], img_size, img_size, img_depth)
        # convert from int to float
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        # rescale values
        # self.x_train /= 255.0
        # self.x_test /= 255.0
        # Save dataset sizes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        # Create one hot array
        self.y_train = to_categorical(self.y_train, 2)
        self.y_test = to_categorical(self.y_test, 2)

    def data_augmentation(self, augment_size=5000): 
        image_generator = ImageDataGenerator(
            rotation_range=10,
            zoom_range = 0.05, 
            width_shift_range=0.07,
            height_shift_range=0.07,
            horizontal_flip=False,
            vertical_flip=False, 
            data_format="channels_last",
            zca_whitening=True)
        # fit data for zca whitening
        image_generator.fit(self.x_train, augment=True)
        # get transformed images
        randidx = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[randidx].copy()
        y_augmented = self.y_train[randidx].copy()
        x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
                                    batch_size=augment_size, shuffle=False).next()[0]
        # append augmented data to trainset
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]

    def next_train_batch(self, batch_size):
        randidx = np.random.randint(self.train_size, size=batch_size)
        epoch_x = self.x_train[randidx]
        epoch_y = self.y_train[randidx]
        return epoch_x, epoch_y
    
    def next_test_batch(self, batch_size):
        randidx = np.random.randint(self.test_size, size=batch_size)
        epoch_x = self.x_test[randidx]
        epoch_y = self.y_test[randidx]
        return epoch_x, epoch_y

    def shuffle_train(self):
        indices = np.random.permutation(self.train_size)
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]

if __name__ == "__main__":
    extracted = True
    if extracted == False:
        extract_cats_vs_dogs(file_dir)
        load_cats_vs_dogs(file_dir)

    if extracted == True:
        data = CATSDOGS(file_dir)
        
        indx = np.random.randint(0, data.train_size, 10)
        imgs = data.x_train[indx]
        labels = data.y_train[indx]

        for img, label in zip(imgs, labels):
            if img_depth == 1:
                plt.imshow(img.reshape((img_size, img_size)))
            else:
                plt.imshow(img)
            plt.title(label)
            plt.show()