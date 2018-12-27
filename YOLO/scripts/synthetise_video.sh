#!/bin/sh

input_image_dir=$1
rate=$2
out_file=$3

ffmpeg -f image2 -i $input_image_dir/image-%8d.jpg -r $rate $out_file
