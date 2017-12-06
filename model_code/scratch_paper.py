  
 
mat=None
with open("./bolbbalgan4/0.npy", "rb") as f:
    mat=np.load(f)

onepiece=mat[0]

print(onepiece.shape)
