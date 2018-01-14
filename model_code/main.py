import argparse
import os
import scipy.misc
import numpy as np
import preprocess as pr
from fin_model import pix2pix
import tensorflow as tf
from glob import glob

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=1024, help='then crop to this size')
#we dont crop
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
#i/o color channels are grayscale, thus #channel == 1
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=False, help='if flip the images for data argumentation')
#we do not flip the data
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
#directory names
parser.add_argument('--dataset_name', dest='dataset_name', default='bolbbalgan4', help='name of the dataset')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--train_tagfile_name', dest='train_tagfile_name', default="tagforfitting.txt", help='training tag file here')
parser.add_argument('--test_tagfile_name', dest='test_tagfile_name', default="tag_evalset_manual.txt", help='test tag file here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')

#loss fucntion weights 
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100, help='weight on L1 term in objective')
parser.add_argument('--L2_lambda', dest='L2_lambda', type=float, default=0, help='weight on L2 term in objective')
parser.add_argument('--GAN_lambda', dest='GAN_lambda', type=float, default=1, help='weight on GAN term in objective')

#generator, discriminator schedule
parser.add_argument('--numD', dest='d_sche', type=int, default=1, help='number of G optim')
parser.add_argument('--numG', dest='g_sche', type=int, default=2, help='number of D optim')
parser.add_argument('--smoothe', dest='smoothe', type=float, default=1, help='generator tanh smoothing')

args = parser.parse_args()

def main(_):
    model_tag="g_{g}_l1_{l1}_l2_{l2}_dg_sche_{dsche}_{gsche}_smoothe_{sm}".format(g=args.GAN_lambda, 
                                                                l1=args.L1_lambda, l2=args.L2_lambda, 
                                                                dsche=args.d_sche, gsche=args.g_sche,
                                                                sm=args.smoothe)
    #logdir
    new_logdir="logs_"+model_tag
    #making directories for 
    new_checkpoint_dir=os.path.join(args.checkpoint_dir, model_tag)
    new_sample_dir=os.path.join(args.sample_dir, model_tag)
    new_test_dir=os.path.join(args.test_dir, model_tag)

    if not os.path.exists(new_checkpoint_dir):
        os.makedirs(new_checkpoint_dir)
    if not os.path.exists(new_sample_dir):
        os.makedirs(new_sample_dir)
    if not os.path.exists(new_test_dir):
        os.makedirs(new_test_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #session = tf.Session(config=config, ...)

    with tf.Session(config=config) as sess:
        if  args.phase=='train': 
            npytrfiles=glob("./{dataset}/*.npy".format(dataset=args.dataset_name))
            if len(npytrfiles)>0: pass
            else: pr.generate_concat_npyfile("./"+args.dataset_name+"/", 
                                    tagfilepath=args.dataset_name+"/"+args.train_tagfile_name) # ./dataset_name is the dir name for the dataset 
        elif args.phase=='test' : pr.generate_v_only_npyfile(args.test_dir, 
                                    tagfilepath=new_test_dir+"/"+args.test_tagfile_name)
        else: exit("--phase argument is only train or test")

        model = pix2pix(sess, image_size=args.fine_size, batch_size=args.batch_size,
                        output_size=args.fine_size, dataset_name=args.dataset_name,
                        checkpoint_dir=new_checkpoint_dir, sample_dir=new_sample_dir, 
                        test_dir=new_test_dir, logdir=new_logdir, d_sche=args.d_sche, g_sche=args.g_sche,
                        smoothe=args.smoothe)


        if args.phase == 'train':
            print("\n\n")
            print("\ttrain")
            print("\tweigt g/l1/l2:{g}, {l1}, {l2}".format(g=args.GAN_lambda, l1=args.L1_lambda, l2=args.L2_lambda))
            print("\tschedule d-g: {dsche},{gsche}".format(dsche=args.d_sche, gsche=args.g_sche))
            print("\ttanh smoothing {sm}".format(sm=args.smoothe))
            print("\n\n")
            model.train(args)
        else:
            model.test(args)

if __name__ == '__main__':
    tf.app.run()
    
