import main_pipe
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


class UndistortTestImages:
    def __init__(self):
        self.test_images_path = 'test_images'
        self.output_images_path = 'output_images'
        self.undistored_test_images_path = ''.join([self.output_images_path, '\\', 'undistored_test_images'])
        self.camera_parameters = 0

    def get_camera_cali_parameters(self):
        with open('camera_cali_parameters.pickle', 'rb') as camera_parameters_file:
            self.camera_parameters = pickle.load(camera_parameters_file)

    def read_test_images(self):
        for raw_image_name in glob.glob(''.join([self.test_images_path, '\\', '*.jpg'])):
            print('processing test image: {}'.format(raw_image_name))
            raw_image = mpimg.imread(raw_image_name)
            self.process_one_test_image(raw_image, raw_image_name)

    def undistort_one_test_image(self, raw_image, raw_image_name):
        test_image_pipe = main_pipe.MainPipeline(raw_image, self.camera_parameters)
        test_image_pipe.pipeline_main()
        pure_filename = raw_image_name[raw_image_name.rfind('\\') + 1:]
        plt.imsave(''.join([self.undistored_test_images_path, '\\', pure_filename]), test_image_pipe.undistorted_image)

    def undistort_test_images_main(self):
        self.get_camera_cali_parameters()
        self.read_test_images()


def main():
    undistort_test_images = UndistortTestImages()
    undistort_test_images.undistort_test_images_main()


if __name__ == '__main__':
    main()