import os
from src.utils import get_project_root

base_path = 'data'
root = get_project_root()


class FeatureClass:
    num_of_elements = 0
    class_path = ''
    img_list = []

    def __init__(self, class_name, input_dir):
        self.class_name = class_name
        classes_dir = os.path.join(base_path, input_dir, class_name)
        self.num_of_elements = len([name for name in os.listdir(classes_dir)
                                    if os.path.isfile(os.path.join(classes_dir, name))])
        # TODO: Load to objects images

    def __str__(self):
        return f"Klasa: {self.class_name} -> {self.num_of_elements} zdjęć"


class FeatureData:
    classes = []

    def __init__(self, input_dir):
        self.child_nbr = None
        self.input_dir = input_dir
        self.load_data()
        # if go to binarize() -> return binarised image
        # if go to original() -> retunr original image
        # if binarize on all object -> make that for all data
        # if binarize on one object -> make that for one image
        # getsize of image
        # return image

    def __str__(self):
        return f"Ilość klas dla ekstrakcji: {self.child_nbr}"

    def load_data(self):
        input_path = os.path.join(base_path, self.input_dir)
        if os.path.isdir(base_path):
            if os.path.isdir(input_path):
                self.classes = [e for e in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, e))]
                if self.classes is not None:
                    for key in self.classes:
                        setattr(self, key, FeatureClass(key, self.input_dir))
                    setattr(self, 'child_nbr', len(self.classes))
            else:
                print(f"Directory: {self.input_dir} is not exist")
        else:
            print(f"Directory: {base_path} is not exist")

    def binarize(self):
        # TODO: object.binarize() -> go through all directories and make binarization
        # TODO: object.class.binarize() -> go through class directory and make binarization
        # TODO: object.class.img.binarize() -> make one image binarization
        # return 0 -> but will create files
        pass

    def getBinarized(self):
        # TODO: object.getBinarized() -> go through all directories and make binarization
        # TODO: object.class.getBinarized() -> go through class directory and make binarization
        # TODO: object.class.img.getBinarized() -> make one image binarization
        # return list of images
        pass
