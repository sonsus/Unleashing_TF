from PIL import Image
import numpy as np
testarray=100*np.random.rand(500,500)
testimg=Image.fromarray(testarray)
print(type(Image.fromarray(testarray)))
#testimg.save("testimg.png")
testimg.save("testimg.jpg")