###### locate this script with its dependencies
###### preprocess.py , wav2spec.py
###### need to be at a parent dir of data

# test generate_concat_npyfile()
# autorun test_writespecgram.py to confirm
import os
import preprocess as pr
import numpy as np
from glob import glob

data_dir="bolbbalgan4_test/"
tagname="tagforfitting.txt"

npyfiles_list=glob("{data_dir}/*npy".format(data_dir=data_dir))
print("\nnum of npyfiles {}".format(len(npyfiles_list)))

if len(npyfiles_list) ==0: pr.generate_concat_npyfile( songdir = data_dir, 
                            tagfilepath= os.path.join(data_dir,tagname) )

for file in npyfiles_list:
    test_ndarray=np.load(file)
    print(test_ndarray.shape)
    test21=test_ndarray.reshape((2048,1024))
    test12=test_ndarray.reshape((1024,2048))
    test_vo=test_ndarray[:,:,0].reshape((1024,1024))
    test_en=test_ndarray[:,:,1].reshape((1024,1024))

    pr.write_specgram_img(test_vo, "{filepath}_vo.jpg".format(filepath=file))
    pr.write_specgram_img(test_en, "{filepath}_en.jpg".format(filepath=file))
# failed below
#    pr.write_specgram_img(test12, "{filepath}_both12_.jpg".format(filepath=file))
#    pr.write_specgram_img(test21, "{filepath}_both21_.jpg".format(filepath=file))