def generate_concat_npyfile(songdir, win_size=win_size,st_size=st_size,tagfilepath=tagfilepath):
    #windowsize and stepsize for chopping wavs. not for specgram
    print("get_shuffled_tr_ex_array")
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


\under construction
#save processed array of shape (?,1024,1024,2) as npy binary file for calling it.
def save_data2npy(name_counter, nparray, save_dir): #one arry per file to utilize load function with ease
    with open("{a}.npy".format(a=name_counter), "wb") as npy:
        np.save(npy,nparray)


\when loading
def loader(filedir):
    res=None
    with open(filedir, "rb") as f:
        res=np.load(f)
    return res
