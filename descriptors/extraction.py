import os
import cv2
import numpy as np
from src.utils import get_project_root
from skimage import io, img_as_ubyte
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import threshold_triangle

base_path = 'data'
root = get_project_root()


class FeatureClass:
    num_of_elements = 0
    class_dir = None
    img_names = []
    img_paths = []
    images = []

    def __init__(self, class_name, input_dir):
        self.class_name = class_name
        self.class_dir = os.path.join(base_path, input_dir, class_name)
        self.num_of_elements = len([name for name in os.listdir(self.class_dir)
                                    if os.path.isfile(os.path.join(self.class_dir, name))])
        self.load_files()
        self.img_paths = [os.path.join(self.class_dir, f) for f in self.img_names]

    def __str__(self):
        return f"Class: {self.class_name} -> {self.num_of_elements} images"

    def load_files(self):
        self.img_names = [file for file in os.listdir(self.class_dir) if file.endswith('.png')]

    def load_images(self):
        self.images = [io.imread(p) for p in self.img_paths]

    def preprocessing(self, output_dir_name):
        for i, img in enumerate(self.images):
            output_dir = os.path.join(base_path, output_dir_name)
            if not os.path.isdir(output_dir):
                print(f"Directory {output_dir_name} not found.")
                os.mkdir(output_dir)
                print("Directory '%s' created" % output_dir_name)

            output_dir_class = os.path.join(output_dir, self.class_name)
            if not os.path.isdir(output_dir_class):
                print(f"Directory {self.class_name} not found.")
                os.mkdir(output_dir_class)
                print("Directory '%s' created" % self.class_name)

            path_img_target = os.path.join(output_dir_class, self.img_names[i])
            if not os.path.isfile(path_img_target):
                img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7)
                img2 = rgb2gray(rgba2rgb(img))
                kernel = np.ones((5, 5), np.uint8)
                img2 = cv2.dilate(img2, kernel, iterations=2)
                thresh = threshold_triangle(img2)
                binary = img2 < thresh
                io.imsave(path_img_target, img_as_ubyte(binary))
                print(f"{path_img_target} ---- DONE")


class FeatureData:
    class_names = []
    is_bin = False

    def __init__(self, input_dir):
        self.classes = []
        self.child_nbr = None
        self.input_dir = input_dir
        self.load_data()
        self.split_test_train()

    def __str__(self):
        return f"Number of classes for extraction: {self.child_nbr}"

    def show(self):
        for i, key in enumerate(self.class_names):
            print(self.__getattribute__(key))

    def load_data(self):
        input_path = os.path.join(base_path, self.input_dir)
        if os.path.isdir(base_path):
            if os.path.isdir(input_path):
                self.class_names = [e for e in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, e))]
                if self.class_names is not None:
                    for key in self.class_names:
                        f_c = FeatureClass(key, self.input_dir)
                        self.classes.append(f_c)
                        setattr(self, key, f_c)
                    setattr(self, 'child_nbr', len(self.class_names))
            else:
                print(f"Directory: {self.input_dir} is not exist")
        else:
            print(f"Directory: {base_path} is not exist")

    def bin(self, output_dir_name):
        print('Binarize process ---- START')
        for class_name in self.classes:
            class_name.load_images()
            class_name.preprocessing(output_dir_name)
        self.is_bin = True
        print('Binarize process ---- DONE')

    def get_bin(self):
        pass
        # img_bin = [io.imread(p) for p in self.img_paths]

    def is_binarized(self):
        return self.is_bin

    def split_test_train(self):
        if self.is_binarized():
            path = os.path.join(root, base_path, 'training')
            img_train_paths, img_test_paths = [], []

            print('Split data process ---- START')
            print(path)
            print('Split data process ---- DONE')

            return path, img_train_paths, img_test_paths
        else:
            print("Make binarization process!")
