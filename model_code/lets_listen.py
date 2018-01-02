import preprocess as pr
import numpy as np 

#this code needs to be run at the same directory with target npyfiles
# ./model_codes

test = ["sample_99.npy", "sample_199.npy"]

for npy_name in test:
    tmp=np.load(npy_name)
    gen=tmp.reshape((1024,1024))
    #recover_audio("./{name}_voice.wav".format(npy_name),voice)
    pr.recover_audio("./{name}_gen.wav".format(name=npy_name),gen)

print("jobs finished")