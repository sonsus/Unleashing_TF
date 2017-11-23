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

# gets list of tuples that has voice range with sec units
def tag2range(wav_name,tagfilepath=tagfilepath):        #wavname contains .wav
    '''
    tagfile must be in form as follows

    --------------tagfile.txt----------------
    wavname1.wav 10-11,20-50
    wavname1_1.wav 0-50
    wavname1_2.wav 0-10
    wavname2.wav 10-11,20-110
    -----------------------------------------

    '''
    namelen=len(wav_name[:-4])#for wav_name contains .wav at the end
    lines=[]
    with open(tagfilepath) as tagfile:
        lines+=tagfile.readlines()
    voice_rangetuples_list=[]
    for line in lines:
        if line[:namelen]==wav_name:
            dash_sep_list=line[namelen+1+4:].rstrip("\n").split(',')   #for windows, \r\n
                                                                # ["10-11","20-110"]
            for dash_sep in dash_sep_list:
                a_range=list(dash_sep.split('-'))
                for i in len(a_range):
                    a_range[i]=int(a_range[i])    #[10,11]
                voice_rangetuples_list.append(tuple(a_range))       

    return voice_rangetuples_list


def get_specgram(rate,filtered_wav):
    #wav obj must underwent bandpass filter    
    specgram = w2s.pretty_spectrogram(filtered_wav.astype('float64'), fft_size = w2s.fft_size, 
                                   step_size = w2s.step_size, log = True, thresh = w2s.spec_thresh)
    row=len(specgram) #this corresponds to time
    col=len(specgram[0])
    if row<col:
        print("\n\n\nNO!\n\n\n")
        sys.exit("sth gone wrong with get_specgram in preprocess.py")
    else:
        specgram=specgram[:col] # specgram is 1024x1024 matrix
    return specgram

def iterative_windower(win_size, st_size, wav, voice_rangetuples_list):
    rate, raw_wav = wavfile.read(wav)
    filtered_wav = w2s.butter_bandpass_filter(raw_wav, w2s.lowcut, w2s.highcut, rate, order=1)
    
    #construct window slinding points from voice_rangetuples_list
    rangelist_set=[]
    for tups in voice_rangetuples_list:
        v_starts=tups[0]
        v_ends=tups[1]
        one_range=np.arange(v_starts,v_ends,st_size)  
        rangelist_set.append(one_range)
    rangelist_set=np.array(rangelist_set)

    #with rangelist_set, chop the filtered_wav
    songpiece_list=[]
    for one_range in rangelist_set:
        start=int(one_range[0]*rate)
        length=int(win_size*rate)              
        if start+length>len(filtered_wav[0]): 
            continue
        else: 
            songpiece=filtered_wav[start:(start+length)]
            songpiece_list.append(songpiece)
    songpiece_array=np.array(songpiece_list)
    return songpiece_array

def get_spec_concat_array(voice_crop_arry, orig_crop_arry):
    spec_concat_list=[]
    for i in len(voice_crop_arry):
        spec_v=get_specgram(voice_crop_arry[i])
        spec_o=get_specgram(orig_crop_arry[i])
        concat_piece=np.concatenate((spec_v,spec_o), axis=1)
        spec_concat_list.append(concat_piece)
    spec_concat_array=np.array(spec_concat_list)
    return spec_concat_array

def get_spec_array(voice_crop_arry):
    spec_vo_list=[]
    for i in len(voice_crop_arry):
        spec_v=get_specgram(voice_crop_arry[i])
        spec_vo_list.append(spec_v)
    spec_array=np.array(spec_vo_list)
    return spec_array

def split2_indiv_spec(spec_concat,select):#if select ="voice" --> returns voice spec, 
    split_result=np.split(spec_concat)
    if select=="voice": res = split_result[0]
    elif select=="ensemble": res = split_result[1]
    return res

def get_shuffled_tr_ex_array(songdir, win_size=win_size,st_size=st_size,tagfilepath=tagfilepath):
    #windowsize and stepsize for chopping wavs. not for specgram
    tr_set_list=[]
    for wav in os.listdir(songdir): #maybe, separated song should be located at lower hierarchy of wav dir
        if wav[0:3]=='vo_': continue
        else: 
            rate_v, raw_v_wav=wavfile.read(songdir+"vo_"+wav)         
            rate_o, raw_o_wav=wavfile.read(songdir+wav)

            voice_rangetuples_list=tag2range(wav,tagfilepath)
            v_songpiece_array=iterative_windower(win_size, st_size, raw_v_wav, voice_rangetuples_list)
            o_songpiece_array=iterative_windower(win_size, st_size, raw_o_wav, voice_rangetuples_list)
            spec_concat_array=get_spec_concat_array(v_songpiece_array, o_songpiece_array)               #this corresponds real AB
            np.random.shuffle(spec_concat_array)

    tr_ex_array=np.array(tr_set_list)
    np.random.shuffle(tr_ex_array) # not sure shuffle here or picking it randomly later 
    return tr_ex_array # thus array is shuffled


#when testing, just voice files to be tested in the directory
#not enough time, thus just copy and paste of get_shuffled_tr_ex_array()
#recommend putting only one file for sake of your mentality
def get_test_vo_ex_array(songdir, win_size=win_size,st_size=st_size*2,tagfilepath=tagfilepath):
    #windowsize and stepsize for chopping wavs. not for specgram
    test_set_list=[]
    for wav in os.listdir(songdir):         #here, filename might be like: vo_somename.wav
        rate, raw_v=wavfile.read(songdir+wav)         
        voice_rangetuples_list=tag2range(wav[3:],tagfilepath)
        v_songpiece_array=iterative_windower(win_size, st_size, raw_v, voice_rangetuples_list)
        spec_concat_array=get_spec_array(v_songpiece_array)               #this corresponds real AB

    test_set_array=np.array(test_set_list) 
    return test_set_array # thus array is shuffled


####### additional but might be quite critical utils #######

#will be used for mode collapse checking
def write_specgram_jpg(specgram, jpgname):   #jpgname with .jpg
    fig, ax = plt.subplots(nrows=1,ncols=1)
    cax = ax.matshow(np.transpose(wav_spectrogram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    #fig.colorbar(cax)
    plt.title('upper: voice only, lower: ensemble')
    plt.savefig(check_training_dir+jpgname,bbox_inches="tight",pad_inches=0)


#takes too much time running. must be used only for testing
def recover_audio(pathandwavname, specgram):
    recovered=w2s.invert_pretty_spectrogram(specgram, fft_size = fft_size,
                                            step_size = step_size, log = True, n_iter = 10)
    #recovered/=max(recovered_audio_orig)
    #recovered*=3                                       #normalize --> amplify.
    wavfile.write(pathandwavname, 44100, recovered)




