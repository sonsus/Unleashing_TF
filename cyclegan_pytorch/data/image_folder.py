###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


'''
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def default_loader(path):
    return Image.open(path).convert('RGB')
'''
#rewrite functions for npy
def is_npy_file(filename):
    return filename.endswith('.npy')

def make_dataset(dir):
    npys=[]
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_npy_file(fname):
                path = os.path.join(root, fname)
                npys.append(path)

    return npys

# opening npyfile
def default_loader(path):
    return np.load(path)

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        npys = make_dataset(root)
        if len(npys) == 0:
            raise(RuntimeError("found 0 npys in {}".format(root)))
            #raise(RuntimeError("Found 0 images in: " + root + "\n"
            #                   "Supported image extensions are: " +
            #                   ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.npys = npys 
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.npys[index] 
        npy = self.loader(path)
        if self.transform is not None:
            npy = self.transform(npy)
        if self.return_paths:
            return npy, path
        else:
            return npy

    def __len__(self):
        return len(self.npys)
