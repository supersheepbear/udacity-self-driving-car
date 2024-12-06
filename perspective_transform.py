import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#plt.ion()


class PerspectiveTransform:
    def __init__(self, image, image_name):
        self.image_name = image_name
        self.image = image
        self.warped_image = 0
        self.output_images_path = 'output_images'
        self.warp_images_path = ''.join([self.output_images_path, '\\', 'warp_test_images'])
        #self.source_points = np.float32([[695, 457], [1092, 719], [206, 719], [585, 457]])
        #self.desired_points = np.float32([[875, 0], [875, 719], [413, 719], [413, 0]])
        #self.source_points = np.float32([[663, 437], [1092, 719], [206, 719], [614, 436]])
        #self.desired_points = np.float32([[875, 0], [875, 719], [413, 719], [413, 0]])
        #self.source_points = np.float32([[663, 437], [1092, 719], [206, 719], [614, 436]])
        #self.source_points = np.float32([[self.image.shape[1] - 580, self.image.shape[0]/2+100],
                                          #[self.image.shape[1] - 200, self.image.shape[0]],
                                          #[200, self.image.shape[0]],
                                          #[580, self.image.shape[0]/2+100]])
        self.source_points = np.float32([[self.image.shape[1] - 580, self.image.shape[0]/2+100],
                                          [self.image.shape[1] - 200, self.image.shape[0]],
                                          [200, self.image.shape[0]],
                                          [580, self.image.shape[0]/2+100]])
        self.desired_points = np.float32([[self.image.shape[1] - 380, 0],
                                          [self.image.shape[1] - 380, self.image.shape[0]],
                                          [380, self.image.shape[0]],
                                          [380, 0]])
        #print(self.source_points)
        #print(self.desired_points)
        self.minv = 0

    def warp_image(self):
        #print('Now warping image: {}'.format(self.image_name))
        image_size = (self.image.shape[1], self.image.shape[0])
        pers_trans_matrix = cv2.getPerspectiveTransform(self.source_points, self.desired_points)
        self.warped_image = cv2.warpPerspective(self.image, pers_trans_matrix, image_size, flags=cv2.INTER_LINEAR)
        self.minv = cv2.getPerspectiveTransform(self.desired_points, self.source_points)

    def save_image(self):
        if not os.path.exists(self.warp_images_path):
            os.makedirs(self.warp_images_path)
        pure_file_name = self.image_name[self.image_name.rfind('\\') + 1:]
        plt.imsave(''.join([self.warp_images_path, '\\', pure_file_name]), self.warped_image)


def plot_source_points():
    img = mpimg.imread('output_images\\undistored_test_images\\straight_lines1.jpg')
    plt.imshow(img)
    size = img.shape
    #plt.plot(695, 457, '.')  # top right
    #plt.plot(1092, 719, '.')  # bottom right
    #plt.plot(206, 719, '.')  # bottom left
    #plt.plot(585, 457, '.')  # top left
    plt.plot(size[1] - 627, size[0]/2+70, '.')  # top right
    plt.plot(size[1] - 200, size[0], '.')  # bottom right
    plt.plot(200, size[0], '.')  # bottom left
    plt.plot(627, size[0]/2+70, '.')  # top left
    a = 1


def pers_trans_test_images():
    for raw_image_name in glob.glob('output_images\\binary_test_images\\*.jpg'):
        image = mpimg.imread(raw_image_name)
        test_image_pers_trans = PerspectiveTransform(image, raw_image_name)
        test_image_pers_trans.warp_image()
        test_image_pers_trans.save_image()


def main():
    pers_trans_test_images()
    #plot_source_points()

if __name__ == '__main__':
    main()