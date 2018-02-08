# code snippet for normalizing specgram

        voice_only=np.reshape(voice_only,(1024,1024))
        ensemble_real=np.reshape(ensemble_real, (1024,1024))
        ensemble_fake=np.reshape(ensemble_fake,(1024,1024)) #np.reshape() returns array ndarray.resize() returns NONE 
        
        #if I wanted non-scaled specgram
        #concat=np.concatenate((voice_only, ensemble_real, ensemble_fake), axis=1)#resulting need to be 1024,3072
        #pr.write_specgram_img()

        # scale ensemble_fake
        max_samples=max(np.absolute(ensemble_fake.flatten("C")))
        max_voice=max(np.absolute(voice_only.flatten("C")))
        ensemble_fake_scaled=ensemble_fake*(max_voice/max_samples)
        # save ensemble_fake as nparray np.load("f.npy will load nparray")
        with open(sample_dir+"/fake_ensemble{a}.npy".format(a=idx), "wb") as f:
            np.save(f,ensemble_fake_scaled)
        
        # write specgram (bot: voice, mid: ensemble_real, top: ensemble_fake)
        normalconcat=np.concatenate((voice_only, ensemble_real, ensemble_fake_scaled), axis=1)
        pr.write_specgram_img(normalconcat, '{}/train_{:02d}_{:06d}.png'.format(sample_dir, epoch, idx))
