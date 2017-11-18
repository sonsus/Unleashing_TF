'''
preprocess.py


input   :   wav file

out     :   list of tr_ex_list (that is matrix) for each song in a directory wav pieces 
            concatenated with its voice counterpart

dependency
        : python 3.5, MFCC.py in the same directory

'''

import os
import numpy as np
import matplotlib.pyplot as plt # for fig saving --> mode collapse check
import wav2spec as w2s # for spectrogram conversion codes
import separation as sep # for voice separation of the original song
                \separation.py is under construction


####parameters#####
win_size=4
st_size=1
#windowing, step size for chopping the specgram. not for making it. beware not to be confused with wav2spec.py


# gets list of tuples that has voice range with sec units
def tag2range(wav_name_with_path, tagfile_with_path ):
    with open("tagfilepath") as tagfile:
        tagfile.readlines()
        parse
        parse
        \under construction 

    return voice_range_list

# get spectrogram of a whole song from a raw song.wav
def get_specgram(rawsong):
    specgram=w2s.pretty_spectrogram(raw_song.astype('float64'), fft_size = w2s.fft_size, 
                                   step_size = w2s.step_size, log = True, thresh = w2s.spec_thresh)
    return specgram

def iterative_windower(win_size, st_size, songwav, voice_rangetuples):
    rate, rawsong = wavfile.read(songwav)
    rangelist_set=[]
    for tups in voice_rangetuples:
        v_starts=tups[0]
        v_ends=tups[1]
        one_range=np.arange(v_starts,v_ends,st_size)  
        rangelist_set.append(one_range)
    rangelist_set=np.array(rangelist_set)

    songpiece_list=[]
    for one_range in rangelist_set:
        start=one_range[0]*rate
        length=st_size*rate
        if start+length>len(rawsong[0]): 
            continue
        else: 
            songpiece=rawsong[start:(start+length)]
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

def get_training_set_matrix(win_size,st_size, voicerange_list, orig_song_wav, voice_song_wav):
    #windowsize and stepsize for chopping wavs. not for specgram
    tr_mat=np.empty()
    for songwavs in os.listdir("mono.wav/files/directory"): #maybe, separated song should be located at lower hierarchy of wav dir
        v_wav, o_wav= sep.sep_voice
        rate_v, raw_v=wavfile.read(songwav_v)           #here I assumed it is imported from w2s
        rate_o, raw_orig=wavfile.read(songwav_o)

        \under construction
        #call seperation.py first to generate voice counterparts
        #open v, and o at the same time
        #iterative_windower() 
        #get_spec_concat_array 
        #append it to the training_set_list
        #training_set_matrix=np.array(training_set_list)
        #****training_set_matrix[0] = first song tr_ex_array
        #****training_set_matrix[0][0] = first song, first piece spectrogram concatenated














