#How our spectrograms are made and fed into the model

#### TL;DR:    
1. pr.generate_concat_npyfile to make npyfiles in bolbbalgan4/, 
2. use np.load(----.npy) to read the data (fin_model.py: load_npy(), )



#### hierarchy that is crucial for model to work    
```
muhan_records/
    bolbbalgan4/
        tagforfitting.txt: all wavfile's tagging (to check that the model fits itself)  
        tag_train_wo_eval.txt: tagging info of only training set (not test set)
    model_code/
        main.py         : trigger, processed files are saved into bolbbalgan4 directory 
        preprocess.py   : open wav files, make it into specgram np.ndarray, and save  
        wav2spec.py     : provides functions for specgram transformation
        fin_model.py    : open saved npy files from main.py at bolbbalgan4/, train/test the model
    test/
        test: test files tagging

```


####details    

> preprocess.py:    
>    wav (1D ndarray) ----(wav2spec.py)---->   
>    filtered signal (= songpiece array, 2D ndarray) ----(wav2spec.py)---->    
>    specgram (2D ndarray with 1 colorchannel = (1024,1024,1) )
>    .npy (binary, ndarray=np.load("\*.npy"))

#####generate_concat_npyfile exploits...   
######  tag2range(): determine what part of the song to be chopped (time tag --> np.ndarray slicing)    
######  iterative_windower    
######  get_specgram()    
######  np.concat    
######  save_data2npy()   
   

   

> main.py: from 75th line to 95th line    
#####1 session is made and check running phase (train or test)
#####2 now pr.generate_concat_npyfile() make specgram into np.ndarray and save its binary as an npyfile
#####3 in case args.phase == "test", data processing pipelines are similar (never ran before. no warranty)


```python
#imported packages and libs below##

import argparse
import os
import scipy.misc
import numpy as np
import preprocess as pr
from fin_model import pix2pix
import tensorflow as tf
from glob import glob

##################################


    with tf.Session(config=config) as sess:
        if  args.phase=='train': 
            npytrfiles=glob("./{dataset}/*.npy".format(dataset=args.dataset_name))
            if len(npytrfiles)>0: pass
            else: 
                pr.generate_concat_npyfile( os.path.join(os.getcwd(),args.dataset_name), 
                                    tagfilepath= os.path.join(args.dataset_name,args.train_tagfile_name) ) # ./dataset_name is the dir name for the dataset 

                #lines of verification code with pr.write_specgram_img() function comes here 
                #you dont need this

        elif args.phase=='test' : pr.generate_v_only_npyfile(args.test_dir, 
                                    tagfilepath= os.path.join(new_test_dir,args.test_tagfile_name) )
        else: exit("--phase argument is only train or test")

```

