import os


class FeatureClass:
    num_of_elements = 0
    class_path = ''
    img_list = []

    def __init__(self, class_name):
        self.class_name = class_name

    def __str__(self):
        return f"Klasa: {self.class_name} -> {self.num_of_elements} zdjęć"


class FeatureData:
    base_path = 'data'
    classes = []

    def __init__(self, input_dir):
        input_path = os.path.join(self.base_path, input_dir)
        if os.path.isdir(self.base_path):
            if os.path.isdir(input_path):
                self.classes = [e for e in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, e))]
                if self.classes is not None:
                    for key in self.classes:
                        setattr(self, key, FeatureClass(key))
                    # setattr(self, 'child_nbr', len(child_names))
            else:
                print(f"Directory: {input_dir} is not exist")
        else:
            print(f"Directory: {self.base_path} is not exist")
        # if go to binarize() -> return binarised image
        # if go to original() -> retunr original image
        # if binarize on all object -> make that for all data
        # if binarize on one object -> make that for one image
        # getsize of image
        # return image

    def load_data(self):
        pass

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
