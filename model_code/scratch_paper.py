at utils.py 
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)





at preprocess.py
#will be used for mode collapse checking
def write_specgram_img(specgram, imgname):   #jpgname with .png
#specgram here has the shape = (1024,1024,2)
    fig, ax = plt.subplots(nrows=1,ncols=1)
    rs_specgram=np.reshape(specgram, (1024,2048))
    cax = ax.matshow(np.transpose(specgram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    #fig.colorbar(cax)
    plt.title('upper: voice only, lower: ensemble')
    plt.savefig(check_training_dir+imgname,bbox_inches="tight",pad_inches=0)