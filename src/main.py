from preprocess import get_data
import numpy as np

def main():
    train_inputs, train_labels, test_inputs, test_labels = get_data('../data/fma_metadata/tracks.csv', '../data/fma_metadata/features.csv')
    print("train labels: ", train_labels)

if __name__ == '__main__':
    main()
