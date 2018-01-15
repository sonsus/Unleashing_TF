# test write specgram img 

import preprocess as pr
import numpy as np
from glob import glob

npyfiles_list=glob("*npy")
print(len(npyfiles_list))

for file in npyfiles_list:
    test_ndarray=np.load(file)
    print(test_ndarray.shape)
    test_vo=test_ndarray[:,:,0].reshape((1024,1024))
    test_en=test_ndarray[:,:,1].reshape((1024,1024))
    print(test_vo == test_en)

    pr.write_specgram_img(test_vo, "vo_{}.jpg".format(file))
    pr.write_specgram_img(test_vo, "en_{}.jpg".format(file))