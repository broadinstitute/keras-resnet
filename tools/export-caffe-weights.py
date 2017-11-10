#!/usr/bin/env python

import caffe
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Export caffe weights to h5 format.")
    parser.add_argument("prototxt", help="Path to prototxt file.")
    parser.add_argument("caffemodel", help="Path to weights file.")
    parser.add_argument("output", help="Path to output weights.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    net = caffe.Net(args.prototxt, caffe.TEST, weights=args.caffemodel)
    net.save_hdf5(args.output)

    print("done.")
