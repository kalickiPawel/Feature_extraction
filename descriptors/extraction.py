import os


class FeatureData:
    def __init__(self, input_dir):
        base_path = 'data/'
        input_path = os.path.join(base_path, input_dir)
        if os.path.isdir(input_path):
            classes = [entry for entry in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, entry))]
            print(classes)
        else:
            print(f"Directory: {input_path} is not exist")
        # set attrib as directory names from data_without_background directory
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
