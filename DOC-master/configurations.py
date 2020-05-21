import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='Process training arguments.')
    parser.add_argument('--nntype', default="Naive", help='The type of the network')
    parser.add_argument('--refpath', help='The path of the reference dataset')
    parser.add_argument('--targetpath', help='The path of the target dataset')
    parser.add_argument('--reflabelpath', '-rlpath', help='The path of the reference labels')
    parser.add_argument('--tarlabelpath', '-tlpath', help='The path of the target labels')


    parser.add_argument('--pretrained', '-pt', help='The type of the pretrained weights')

    parser.add_argument('--batches', '-bs', type=int, default=2, help='number of batches')
    parser.add_argument('--epochs', '-ep', type=int, default=20, help='number of epochs')
    parser.add_argument('--optimizer', '-opt', default="adam", help='optimizer  type')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--lambd', type=float, default=0.1, help='lambda constant, the impact of the compactness loss')


    parser.add_argument('--output_path', default=os.getcwd(), help='The path to keep the output')
    return parser.parse_args()