#!/usr/bin/env python
# coding: utf-8
import sys
import argparse

parser = argparse.ArgumentParser(description='Neural Style Transfer')
parser.add_argument('--content_image', '-c', help="input content image path")
parser.add_argument('--style_image', '-s', help="input style image path")
parser.add_argument('--iteration_number', '-i', type=int, help="iteration number")
args = parser.parse_args()
if len(sys.argv) <= 1:
    parser.print_help()
    exit(1)

from NST import *

main(args.content_image, args.style_image, args.iteration_number)
