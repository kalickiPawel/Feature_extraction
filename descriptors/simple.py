from skimage import io

import numpy as np
import cv2
import os


class SimpleDescriptors:
    dataset_path = ""
    train_set_paths, test_set_paths = [], []
    train_set, test_set = [], []
    results = []

    def __init__(self, train_p, test_p, data_obj):
        self.class_names = sorted(data_obj.class_names)
        self.train_set_paths, self.test_set_paths = train_p, test_p
        self.prepare_img()
        print(f"Number of Train feature: {len(self.train_set)}")
        print(f"Number of Test feature: {len(self.test_set)}")

    def prepare_img(self):
        for train in self.train_set_paths:
            d = {'class_name': os.path.basename(os.path.dirname(train)), 'file': io.imread(train)}
            self.train_set.append(d)
        for test in self.test_set_paths:
            d = {'class_name': os.path.basename(os.path.dirname(test)), 'file': io.imread(test)}
            self.test_set.append(d)

    def fit(self):
        train = [f['file'] for f in self.train_set]
        train_learned = []
        area, per, round, compact, convex = [], [], [], [], []

        for i, item in enumerate(train):
            area.append(self.area_desc(item))
            per.append(self.perimeter_desc(item))
            round.append(self.roundness_desc(item))
            compact.append(self.compactness_desc(item))
            convex.append(self.convex_hull_desc(item))

        train_learned.append(area)
        train_learned.append(per)
        train_learned.append(round)
        train_learned.append(compact)
        train_learned.append(convex)
        return train_learned

    @staticmethod
    def area_desc(img):
        return np.count_nonzero(img == 255)

    @staticmethod
    def perimeter_desc(img):
        edged = cv2.Canny(img, 30, 200)
        # contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # plt.imshow(edged, cmap='gray')
        # plt.show()
        return np.count_nonzero(edged == 255)

    @staticmethod
    def roundness_desc(img):
        area = np.count_nonzero(img == 255)
        edged = cv2.Canny(img, 30, 200)
        perimeter = np.count_nonzero(edged == 255)
        return np.power(perimeter, 2) / (4 * np.pi * area)

    @staticmethod
    def compactness_desc(img):
        area = np.count_nonzero(img == 255)
        edged = cv2.Canny(img, 30, 200)
        perimeter = np.count_nonzero(edged == 255)
        return np.power(perimeter, 2) / area

    @staticmethod
    # https://www.learnopencv.com/convex-hull-using-opencv-in-python-and-c/
    def convex_hull_desc(img):
        ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull = []
        # calculate points for each contour
        for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i], False))
        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
        for i in range(len(contours)):
            color_contours = (0, 255, 0)
            color = (255, 0, 0)
            # draw ith contour
            cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
            # draw ith convex hull object
            cv2.drawContours(drawing, hull, i, color, 1, 8)
        # plt.imshow(drawing, cmap=plt.cm.gray)
        # plt.show()
        edged = cv2.Canny(drawing, 30, 200)
        return np.count_nonzero(edged == 255)

    def test(self, clf):
        results = []
        for img in self.test_set:
            results.append(self.classify(img, clf))
        return results

    def classify(self, img, train):
        img_file = img['file']

        tmp = [
            self.area_desc(img_file),
            self.perimeter_desc(img_file),
            self.roundness_desc(img_file),
            self.compactness_desc(img_file),
            self.convex_hull_desc(img_file)
        ]
        res = [self.enc_idx(self.find_nearest(train[i], tmp[i])) for i in range(5)]

        print("Obraz wejściowy: ", img['class_name'])
        print("Deskryptor - pole powierzchni: ", res[0])
        print("Deskryptor - obwód: ", res[1])
        print("Deskryptor - kołowość: ", res[2])
        print("Deskryptor - zwartość: ", res[3])
        print("Deskryptor - obwód powłoki wypukłej: ", res[4])

        out = []
        for i in range(5):
            out.append(1) if res[i] == img['class_name'] else out.append(0)

        return out

    def enc_idx(self, i):
        return self.class_names[i]

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    @staticmethod
    def score(scores):
        score_types = ['area', 'peri', 'roun', 'comp', 'conv']
        results = {el: 0 for el in score_types}
        for i in range(len(scores)):
            for j, result in enumerate(results):
                results[result] += scores[i][j]

        for s in score_types:
            results[s] = results[s] / len(scores) * 100

        return list(results.values())
