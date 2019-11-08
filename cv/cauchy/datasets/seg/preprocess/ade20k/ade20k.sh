#!/bin/bash

cd ~/data
rm -rf ade20k

mkdir ade20k

cd ade20k

mkdir -p images/training annotations/training images/validation annotations/validation

cp -r ~/data/ade20k_pre/train/image/*.jpg images/training 
cp -r ~/data/ade20k_pre/train/label/*.png annotations/training 

cp -r ~/data/ade20k_pre/val/image/*.jpg images/validation
cp -r ~/data/ade20k_pre/val/label/*.png annotations/validation 