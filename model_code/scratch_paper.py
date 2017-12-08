from numpy import *

a=array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
b=zeros(a.shape)
c=arange(20).reshape(a.shape)
print("c is")
print(c)

print("\n\nfor np.sum()")
print("sum over axis 3? possible, 4? no")
for i in range(3):
    test=sum((a,c), axis=i)
    print(test)



print("\n\nfor concat")
print("concat over axis 3? N.O.")
for i in range(3):    
    temp=concatenate((a,b),axis=i)
    print(temp)
