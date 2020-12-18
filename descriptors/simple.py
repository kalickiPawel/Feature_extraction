from skimage import io


class SimpleDescriptors:
    dataset_path = ""
    train_set_paths, test_set_paths = [], []
    train_img, test_img = [], []

    def __init__(self, path, train_p, test_p):
        self.dataset_path = path
        self.train_set_paths, self.test_set_paths = train_p, test_p
        self.train_img, self.test_img = self.prepare_img()
        print(f"Number of Train feature: {len(self.train_img)}")
        print(f"Number of Test feature: {len(self.test_img)}")

    def prepare_img(self):
        return [io.imread(p) for p in self.train_set_paths], [io.imread(p) for p in self.test_set_paths]
