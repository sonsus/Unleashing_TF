# Unleashing limit project log 1: pix2pix
(continued on log2: //https://github.com/sonsus/generative-models/blob/master/README.md)
# pix2pix for generating karaoke sound (for given a vocal)
## blueprint
1. GAN seems to synthesize fake faces quite well
2. even conditional generation looks possible
3. how bout applying it to the spectrogram ? (not an acoustic vector)

## control 
- it might not be just a magic, we need coherency or at least patterns learnable in the data
- picked bolbbalgan4 (singer/writer) whose music shares similar vibe

## data preprocessing
### voice extraction
- J. L. Durrieu Journal on Selected Topics on Signal Processing 2011 https://github.com/sonsus/separateLeadStereo
- not perfect but seemed usable
### cropping the songs
- voiceless parts were abandoned
- cropped with 1sec windowing
- to make the input square-shaped, 1024 timestep (=3.x secs) window is chosen
### Spectrogram transform
- FFT with 1024 bin bunch of constants written in the code

## exp setting
### Based on pix2pix (P.Isola CVPR2016)
1. training
- follow alternative training schema of standard GANs
- voice spectrogram piece is fed as a condition (or a tag) for ensemble spectrogram
- hope generator learns to figure out how to synth the ensemble from voices

2. test
- raw input is not normalized as well as noisy in other way to the extracted ones
- both are noisy and each noise is differ from each other: cannot use raw voice as an input
- use voice clips extracted (not in training set)


## result
- seems like generator makes rational image of spectrograms at a first glance
- it starts getting worse and never getting back
- failed even for training set fitting
