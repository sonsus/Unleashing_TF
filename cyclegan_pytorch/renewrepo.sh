#this script need to be located at parent directory of pytorch git

timestamp() {
  date +"%T"
}

mv pytorch-CycleGAN-and-pix2pix/datasets/bolbbalgan4 .
mv muhan_records/model_code/checkpoints .

echo "cloning"
timestamp
rm -rf pytorch-CycleGAN-and-pix2pix
git clone https://github.com/sonsus/pytorch-CycleGAN-and-pix2pix


echo "remove"
timestamp
rm -rf pytorch-CycleGAN-and-pix2pix/datasets/bolbbalgan4
rm -rf pytorch-CycleGAN-and-pix2pix/checkpoints

echo "move"
timestamp
mv bolbbalgan4 pytorch-CycleGAN-and-pix2pix/datasets/ 
mv checkpoints pytorch-CycleGAN-and-pix2pix

