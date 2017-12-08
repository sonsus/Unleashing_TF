'''
preprocess.py                     

makes tr_set_mat which is 2D nparray

tr_set_mat contains tr_ex_array which is 1D nparray
tr_ex_array contains spec_concat_array which is concatenated spectrograms of voice and original wav (2D nparray)

        ****tr_set_mat[0] = first song tr_ex_array
        ****tr_set_mat[0][0] = first song, first piece's concatenated spectrogram from voice and original

recommend to use easy names for each songs(with eng)
songs (either voice-only and originals) should be placed in the same directory


**this also contains
    write_specgram_jpg(specgram, jpgname)   #jpgname with .jpg
    recover_audio(pathandwavname, specgram)

**wav2spec.py need to be placed in the same directory that preprocess.py, and the model scripts exist



'''

import os
import numpy as np
import matplotlib.pyplot as plt # for fig saving --> mode collapse check
import wav2spec as w2s # for spectrogram conversion codes
from scipy.io import wavfile
import sys
# import separation as sep # for voice separation of the original song
                #no need to do this. just prep voice separated files before.


####parameters#####
#windowing, step size for chopping the specgram.
win_size=4
st_size=1           #also float available

songdir="monowav/files/directory/"      #beware this must contain "/"
tagfilepath="where/exists/tagfile.txt"  
check_training_dir="where/specgram/jpg/are/stored/"     #for checking mode collapse

'''
    tagfile must be in form as follows

    --------------tagfile.txt----------------
    wavname1.wav 10-11,20-50
    wavname1_1.wav 0-50
    wavname1_2.wav 0-10
    wavname2.wav 10-11,20-110
    -----------------------------------------

'''

# gets list of tuples that has voice range with sec units
def tag2range(wav_name,tagfilepath=tagfilepath):        #wavname contains .wav
    print("tag2range")
    namelen=len(wav_name[:-4])#for wav_name contains .wav at the end
    lines=[]
    with open(tagfilepath) as tagfile:
        lines+=tagfile.readlines()
#        print(lines)
#        print("         namelen=%s"%namelen)
    voice_rangetuples_list=[]
    for line in lines:
        if line[:namelen]==wav_name[:-4]: 
            dash_sep_list=line[namelen+1+4:].rstrip("\n").split(',')   
#            print(dash_sep_list)
            for dash_sep in dash_sep_list:
                a_range=list(dash_sep.split('-'))
                for i in range(len(a_range)):
                    a_range[i]=int(a_range[i])    #[10,11]
                voice_rangetuples_list.append(tuple(a_range))       
#    print("voice_rangetuples_list is")
#    print(voice_rangetuples_list)
    return voice_rangetuples_list


def get_specgram(rate,filtered_wav):
    #wav obj must underwent bandpass filter
    #print("get_specgram")    
    specgram = w2s.pretty_spectrogram(filtered_wav.astype('float64'), fft_size = w2s.fft_size, 
                                   step_size = w2s.step_size, log = True, thresh = w2s.spec_thresh)
    row=len(specgram) #this corresponds to time
    col=len(specgram[0])
    if row<col:
        print("\n\n\nNO!\n\n\n")
        sys.exit("sth gone wrong with get_specgram in preprocess.py")
    else:
        specgram=specgram[:col] # specgram is 1024x1024 matrix
    #print(specgram.shape)
    return specgram

def iterative_windower(win_size, st_size, wav, voice_rangetuples_list):
    print("iterative_windower")
    rate, raw_wav = wavfile.read(wav)
    filtered_wav = w2s.butter_bandpass_filter(raw_wav, w2s.lowcut, w2s.highcut, rate, order=1)
    
    #construct window slinding points from voice_rangetuples_list
    rangelist_set=[]
    for tups in voice_rangetuples_list:
        v_starts=tups[0]
        v_ends=tups[1]
        one_range=np.arange(v_starts,v_ends,st_size)
        print("onerange=%s"%one_range)  
        rangelist_set.append(one_range)
    rangelist_set=np.array(rangelist_set)

    #with rangelist_set, chop the filtered_wav
    songpiece_list=[]
    for one_range in rangelist_set:
        for stpt in one_range:
            if (filtered_wav.shape[0]-stpt*rate) < win_size*rate: continue
            else: 
                songpiece=filtered_wav[stpt*rate:(stpt+win_size)*rate]
                songpiece_list.append(songpiece)
    songpiece_array=np.array(songpiece_list)
    print(songpiece_array.shape)
    return rate, songpiece_array

def get_spec_concat_array(rate_v, rate_o, voice_crop_arry, orig_crop_arry):
    print("get_spec_concat_array")
    spec_concat_list=[]
    for i in range(len(voice_crop_arry)):
        spec_v=get_specgram(rate_v, voice_crop_arry[i])
        spec_o=get_specgram(rate_o, orig_crop_arry[i])
        rs_spec_v=np.reshape(spec_v,(1024,1024,1))
        rs_spec_o=np.reshape(spec_o,(1024,1024,1))
        concat_piece=np.concatenate((rs_spec_v,rs_spec_o), axis=1)
        spec_concat_list.append(concat_piece)
    spec_concat_array=np.array(spec_concat_list)
    print(spec_concat_array.shape)
    return spec_concat_array

def get_spec_array(rate, voice_crop_arry):
    print("get_spec_array")
    spec_vo_list=[]
    for i in len(voice_crop_arry):
        spec_v=get_specgram(rate, voice_crop_arry[i])
        rs_spec_v=np.reshape(spec_v,(1024,1024,1))
        spec_vo_list.append(spec_v)
    spec_array=np.array(spec_vo_list)
    print(spec_array.shape)
    return spec_array

def split2_indiv_spec(spec_concat,select):#if select ="voice" --> returns voice spec, 
    split_result=np.split(spec_concat)
    if select=="voice": res = split_result[0]
    elif select=="ensemble": res = split_result[1]
    return res

def save_data2npy(name_counter, nparray, save_dir): #one arry per file to utilize load function with ease
    with open("{a}.npy".format(a=name_counter), "wb") as npy:
        np.save(npy,nparray)

def generate_concat_npyfile(songdir, win_size=win_size,st_size=st_size,tagfilepath=tagfilepath):
    #windowsize and stepsize for chopping wavs. not for specgram
    print("generate_concat_npyfile")
    counter=0
    for wav in os.listdir(songdir): #maybe, separated song should be located at lower hierarchy of wav dir
        if wav[0:3]=='vo_': continue
        else: 
            #rate_v, raw_v_wav=wavfile.read(songdir+"vo_"+wav)
            #rate_o, raw_o_wav=wavfile.read(songdir+wav)
            #print(raw_v_wav.shape)
            #print(raw_o_wav.shape)
            voice_rangetuples_list=tag2range(wav,tagfilepath)
            rate_v, v_songpiece_array=iterative_windower(win_size, st_size, songdir+"vo_"+wav, voice_rangetuples_list)
            rate_o, o_songpiece_array=iterative_windower(win_size, st_size, songdir+wav, voice_rangetuples_list)
            spec_concat_array=get_spec_concat_array(rate_v, rate_o, v_songpiece_array, o_songpiece_array)               #this corresponds real AB
            save_data2npy(name_counter=counter, nparray=spec_concat_array, save_dir=songdir)
            counter+=1 

'''memory does not allow to use this function
def get_shuffled_tr_ex_array(songdir, win_size=win_size,st_size=st_size,tagfilepath=tagfilepath):
    #windowsize and stepsize for chopping wavs. not for specgram
    print("get_shuffled_tr_ex_array")
    counter=0
    tr_ex_array=None
    for wav in os.listdir(songdir): #maybe, separated song should be located at lower hierarchy of wav dir
        if wav[0:3]=='vo_': continue
        else: 
            #rate_v, raw_v_wav=wavfile.read(songdir+"vo_"+wav)
            #rate_o, raw_o_wav=wavfile.read(songdir+wav)
            #print(raw_v_wav.shape)
            #print(raw_o_wav.shape)
            voice_rangetuples_list=tag2range(wav,tagfilepath)
            rate_v, v_songpiece_array=iterative_windower(win_size, st_size, songdir+"vo_"+wav, voice_rangetuples_list)
            rate_o, o_songpiece_array=iterative_windower(win_size, st_size, songdir+wav, voice_rangetuples_list)
            spec_concat_array=get_spec_concat_array(rate_v, rate_o, v_songpiece_array, o_songpiece_array)               #this corresponds real AB
            if counter==0: tr_ex_array=spec_concat_array
            else: tr_ex_array=np.concatenate((tr_ex_array,spec_concat_array),axis=0)
            counter+=1
    np.random.shuffle(tr_ex_array) # not sure shuffle here or picking it randomly later 
    print("resulted imageset is,")
    print(tr_ex_array.shape)
    return tr_ex_array # thus array is shuffled
'''

def generate_v_only_npyfile(songdir, win_size=win_size,st_size=st_size*2,tagfilepath=tagfilepath):
    #this function is almost twin with generate_concat_npyfile
    #songdir=testdir with only vo_somename.wav files 
    counter=0
    for wav in os.listdir(songdir):         
        rate, raw_v=wavfile.read(songdir+wav)
        voice_rangetuples_list=tag2range(wav[3:],tagfilepath)
        rate_v, v_songpiece_array=iterative_windower(win_size, st_size, wav, voice_rangetuples_list)
        spec_v_array=get_spec_array(rate_v, v_songpiece_array)               #this corresponds real AB
        save_data2npy(name_counter=counter, nparray=spec_v_array, save_dir=songdir)
        counter+=1

'''memory didnt allowed it!
#when testing, just voice files to be tested in the directory
#not enough time, thus just copy and paste of get_shuffled_tr_ex_array()
#recommend putting only one file for sake of your mentality
def get_test_vo_ex_array(songdir, win_size=win_size,st_size=st_size*2,tagfilepath=tagfilepath):
    #windowsize and stepsize for chopping wavs. not for specgram
    counter=0
    test_set_array=None
    for wav in os.listdir(songdir):         #here, filename might be like: vo_somename.wav
        rate, raw_v=wavfile.read(songdir+wav)
        voice_rangetuples_list=tag2range(wav[3:],tagfilepath)
        rate_v, v_songpiece_array=iterative_windower(win_size, st_size, wav, voice_rangetuples_list)
        spec_v_array=get_spec_array(rate_v, v_songpiece_array)               #this corresponds real AB
        if counter==0: test_set_array=spec_v_array
        else: tr_ex_array=np.concatenate((test_set_array,spec_v_array),axis=0)
        counter+=1
    print("resulted testset is")
    print(test_set_array.shape)
    return test_set_array # thus array is shuffled
'''

####### additional but might be quite critical utils #######

#will be used for mode collapse checking
def write_specgram_img(specgram, imgname):   #jpgname with .png
#specgram here has the shape = (1024,1024,2)
    fig, ax = plt.subplots(nrows=1,ncols=1)
    rs_specgram=np.reshape(specgram, (1024,2048))
    cax = ax.matshow(np.transpose(rs_specgram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    #fig.colorbar(cax)
    plt.title('upper: voice only, lower: ensemble')
    plt.savefig(imgname,bbox_inches="tight",pad_inches=0)
    #plt.savefig(check_training_dir+imgname,bbox_inches="tight",pad_inches=0)


#takes too much time running. must be used only for testing
def recover_audio(pathandwavname, specgram):
    rs_specgram=np.reshape(specgram, (1024,1024))
    recovered=w2s.invert_pretty_spectrogram(rs_specgram, fft_size = fft_size,
                                            step_size = step_size, log = True, n_iter = 10)
    #recovered/=max(recovered_audio_orig)
    #recovered*=3                                       #normalize --> amplify.
    wavfile.write(pathandwavname, 44100, recovered)


#save processed array of shape (?,1024,1024,2) as npy binary file for calling it.
def save_data2npy(name_counter, nparray, save_dir): #one arry per file to utilize load function with ease
    with open("{path}{a}.npy".format(path=save_dir, a=name_counter), "wb") as npy:
        np.save(npy,nparray)

'''moved to utils.py as load_npy()
def loader(filedir):
    res=None
    with open(filedir, "rb") as f:
        res=np.load(f)
    return res
'''

