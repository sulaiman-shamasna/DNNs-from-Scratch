import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size #[height, width , channels]
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.img_ptr = 0
        self.data_size = 100

        remainder = np.remainder(self.data_size, self.batch_size)
        self.class_index = np.arange(self.data_size)

        self.all_images = []
        for idx in range(self.data_size):
            i = np.load(file_path + str(idx) + '.npy')
            self.all_images.append(np.asarray(i))
        with open(self.label_path, 'r') as f:
            self.all_labels = json.load(f)

        if remainder > 0:
            for jj in range(self.batch_size-remainder):
                i = np.load(self.file_path + str(jj)+ '.npy')
                self.all_images.append(np.asarray(i))
                self.class_index = np.append(self.class_index,jj)
            self.data_size += (self.batch_size - remainder)
        self.all_images = np.array(self.all_images)

        self.start_new_epoch = True # flag to indicate start of new epoch

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        from skimage import transform

        resized_images = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]))
        if self.start_new_epoch is True:
            self.new_all_images = np.copy(self.all_images).astype(np.uint8)
        # shuffle only once per epoch, set flag to false so its not shuffled everytime
        # if in the middle of the epoch, just reuse the self.new_all_images that was created at the beginning of the epoch
        if self.shuffle is True and self.start_new_epoch is True:
            self.start_new_epoch = False
            np.random.shuffle(self.class_index)
            self.new_all_images = self.all_images[self.class_index]

        # To send only one batch at a time
        # resize that batch to resized images
        if (self.image_size[0] != self.all_images[0].shape[0]) or (self.image_size[1] != self.all_images[0].shape[1]):
            ##numpy resize can be used to avoid the for loop, and do the entire 4D array in one step, but np.resize does not
            ##interpolate, it copies from the beginning rows and appends at the bottom...
            ##this is OK for this exercise? but not what is needed later on.
            for idx in range(self.batch_size):
                resized_images[idx] = (transform.resize(self.new_all_images[self.img_ptr+idx], self.image_size) * 255).astype(
                    np.uint8)
        else:
            resized_images = np.copy(self.new_all_images[self.img_ptr:self.img_ptr + self.batch_size])

        labels = np.zeros(self.batch_size).astype(int)
        for idx in range(self.batch_size):
            labels[idx] = self.all_labels[str(self.class_index[self.img_ptr + idx])]

        #call mirror/rotate randomly for some images in the batch if set true
        if self.mirroring is True or self.rotation is True:
            i = np.random.randint(self.batch_size)
            rand_idx = np.random.randint(self.batch_size, size = i)
            for idx in range(len(rand_idx)):
                resized_images[rand_idx[idx]] = self.augment(resized_images[rand_idx[idx]])


        #Update the pointer for next batch
        #do not update when only labels are asked for?
        self.img_ptr += self.batch_size
        if self.img_ptr < self.data_size:
            #make sure to set the flag to false , so next time the data is not shuffled
            self.start_new_epoch = False
        elif self.img_ptr == self.data_size:
            self.img_ptr = 0
            #set the flag = True to start new epoch when next() is called again
            self.start_new_epoch = True # start new epoch

        #self.show() #debug step
        return resized_images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if self.mirroring is True:
            img = np.flip(img, axis=1)

        if self.rotation is True:
            rot_by = np.random.randint(3)
            for idx in range(rot_by+1):
                img = np.rot90(img)

        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        class_name = self.class_dict[x]
        return class_name

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        batch, batch_labels = self.next()

        mm = 3 ## number of columns for the subplots, its fixed here
        nn = int(np.round(self.batch_size/mm)) ## required number of rows in the subplot
        fig, axs = plt.subplots(mm,nn)
        for i in range(mm):
            for j in range(nn):
                if i*nn+j == self.batch_size:
                    break
                axs[i,j].imshow(batch[i*(nn)+j].astype(np.uint8))
                axs[i,j].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
                cname = self.class_name(batch_labels[i*nn+j])
                axs[i,j].xaxis.set_label_position('top')
                axs[i,j].get_yaxis().set_visible(False)
                axs[i,j].set_xlabel(cname)

        plt.show()
        return
