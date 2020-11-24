from preprocess import get_data
import numpy as np

def main():
    train_inputs, train_labels = get_data('../data/fma_metadata/tracks.csv', '../data/fma_metadata/features.csv')

if __name__ == '__main__':
    main()
