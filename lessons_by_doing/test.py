#test.py
import numpy as np
emp=np.array([])

a=np.array([[1,2],
            [3,4]])
a1=a*2
a_=np.concatenate((a,a1),axis=1)

b=-1*np.array([[1,2],
            [3,4]])
b1=b*2
b_=np.concatenate((b,b1),axis=1)


print(a)
print(a1)
print(a1[:,:,None])
print(a_)

s0=np.split(a_,2, axis=0)
s1=np.split(a_,2, axis=1)
print(s0)
print(s1)

'''
zero=np.zeros((2,2))
one=np.ones((2,2))
twos=2*one
three=3*one
four=4*one
five=5*one


print(a_)
print(b_)
res=[]
res.append(a_)
res.append(b_)
res.append(one)
res.append(twos)
res.append(three)
res.append(four)
res.append(five)

print(res)
np.random.shuffle(res)
print(res)
'''
#c=np.concatenate((a,b),axis=1)



#print(c)
#print(np.transpose(c))
'''
d=np.append(a,b)

print(c)
print(d)
    
b=np.arange(2,5,0.5)
c=[a,b]
print(type(c))
npc=np.array(c)
print(type(npc))
print(type(npc[0]))

'''