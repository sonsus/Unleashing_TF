from numpy import *

set_printoptions(threshold=nan)


a=array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
b=arange(start=0, step=2, stop=40).reshape((5,4))

c=stack((a,b))
print(a)
print(b)
print(c)

'''
a_=a[:,:]
a__=a[:,:,None,None]
print(a__)
print(a_)

print("a_.shape={s}".format(s=a_.shape))
print("a__.shape={s}".format(s=a__.shape))

'''
'''
def get_spec_concat_npy(rate_v, rate_o, voice_crop_arry, orig_crop_arry, song_no, savedir):
    for piece_no in range(len(voice_crop_arry)):
        spec_v=get_specgram(rate_v, voice_crop_arry[i])
        spec_o=get_specgram(rate_o, orig_crop_arry[i])
        rs_spec_v=np.reshape(spec_v,(1024,1024,1))
        rs_spec_o=np.reshape(spec_o,(1024,1024,1))
        concat_piece=np.concatenate((rs_spec_v,rs_spec_o), axis=1)
        save_data2npy(name_counter=song_no+piece_no,nparray=concat_piece,savedir=savedir)


def generate_concat_npyfile(songdir, win_size=win_size,st_size=st_size,tagfilepath=tagfilepath):
    #windowsize and stepsize for chopping wavs. not for specgram
    print("generate_concat_npyfile")
    for i, wav in enumerate(os.listdir(songdir)): #maybe, separated song should be located at lower hierarchy of wav dir
        if wav[0:3]=='vo_': continue
        else: 
            voice_rangetuples_list=tag2range(wav,tagfilepath)
            rate_v, v_crop_arry=iterative_windower(win_size, st_size, songdir+"vo_"+wav, voice_rangetuples_list)
            rate_o, o_crop_arry=iterative_windower(win_size, st_size, songdir+wav, voice_rangetuples_list)
            get_spec_concat_npy(rate_v, rate_o, v_crop_arry, o_crop_arry, song_no=i*10000, savedir=songdir)  

def save_data2npy(name_counter, nparray, save_dir): #one arry per file to utilize load function with ease
    with open("{a}.npy".format(a=name_counter), "wb") as npy:
        np.save(npy,nparray)






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
'''
