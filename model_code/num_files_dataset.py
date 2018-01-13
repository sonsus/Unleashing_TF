#count files in dataset folder
#run this in parent folder of dataset folder (e.g. parent/dataset --> run at parent )

import os
pathname=os.path.join(os.getcwd(),"bolbbalgan4")
files_list = os.listdir(os.path.join(os.getcwd(),"bolbbalgan4"))
numfiles=len(files_list)
print("num of files in dataset(={pathname}) is:".format(pathname=pathname))
print("\t{num}".format(num=numfiles))