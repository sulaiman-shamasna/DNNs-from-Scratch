import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, size, tilesize):
        #Size = total size of the checkerboard,
        #tilesize = size of each tile
        #here we create a single tile and then add/concatenate to it later
        self.size=size
        self.tilesize=tilesize
        self.output = np.zeros((tilesize, tilesize))

    def draw(self):
        #first we create a 2x2 tile block and then simply repeat it n times in horizontal and vertical direction
        #check that size can be divided by 2*tilesize
        num_tiles = int(self.size / (2 * self.tilesize))
        if (np.remainder(self.size, 2*self.tilesize) != 0):
            print("total size is not a multiple of 2*tilesize, exiting draw method")
            return
        self.output = np.concatenate((self.output, np.ones((self.tilesize,self.tilesize))), axis=1)
        temp_output = np.concatenate((np.ones((self.tilesize, self.tilesize)), np.zeros((self.tilesize, self.tilesize))), axis=1)
        self.output = np.concatenate((self.output, temp_output), axis=0)
        self.output = np.tile(self.output, (num_tiles, num_tiles))
        copy_output = np.copy(self.output)

        return copy_output

    def show(self):
        if (np.remainder(self.size, 2*self.tilesize) != 0):
            print("total size is not a multiple of 2*tilesize, exiting show method")
            return
        plt.matshow(self.output, cmap='gray')
        plt.show()
        return


class Circle:
    def __init__(self, resolution, radius, position):

        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = np.zeros((resolution, resolution))

    def draw(self):

        X, Y = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
        X = abs(X-self.position[0])**2
        Y = abs(Y-self.position[1])**2
        Z = X + Y
        mask = Z<self.radius**2
        self.output = np.logical_or(self.output, mask)
        copy_output = np.copy(self.output)
        return copy_output

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()
        return

class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.zeros((resolution, resolution, 3))

    def draw(self):
        self.output.astype(float)
        self.output[:,:,0] = np.linspace(0,1, self.resolution)
        self.output[:,:,1] = np.array([np.linspace(0,1, self.resolution),]*self.resolution).transpose()
        self.output[:,:,2] = np.linspace(1,0, self.resolution)
        copy_output = np.copy(self.output)
        return copy_output

    def show(self):
        plt.imshow(self.output)
        plt.show()
        return
