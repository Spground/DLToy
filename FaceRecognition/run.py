#!/usr/bin/env python
# coding: utf-8
import sys
import argparse

parser = argparse.ArgumentParser(description='Face Recognition')
parser.add_argument('--database_dir', '-d', help="your face database dir")
parser.add_argument('--input_face', '-i', help="input face image path")
args = parser.parse_args()
if len(sys.argv) < 2:
    parser.print_help()
    exit(1)

from FaceRecognition import *

main(args.database_dir, args.input_face)
