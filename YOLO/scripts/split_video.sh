#!/bin/sh
rate=$2
input_video=$1
out_dir=$3

echo input_video: $input_video
echo sample_rate:$rate
echo output dir:$out_dir

if ! test -f $input_video
then
    echo input video : $input_video does not exits.
    exit 1
else
    ffmpeg -i $input_video -f image2 -r $rate $out_dir/image-%8d.jpg
fi

echo done.
