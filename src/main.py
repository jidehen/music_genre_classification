import preprocess
from load_audio import walk_files, read_from_npy
import numpy as np
import tensorflow as tf
from genre_model import Model
from seq2seq import RNN_Seq2Seq
import matplotlib.pyplot as plt

def train(model, numerical_inputs, feature_inputs, char_inputs, train_labels):
    max_itr = len(feature_inputs) // model.batch_size

    losses = []
    for i in range(max_itr):
        print("Batch " + str(i+1) + " / " + str(max_itr))
        numerical_input_batch = preprocess.get_batch(numerical_inputs, i*model.batch_size, model.batch_size)
        feature_input_batch = preprocess.get_batch(feature_inputs, i*model.batch_size, model.batch_size)
        char_input_batch = preprocess.get_batch(char_inputs, i*model.batch_size, model.batch_size)
        label_batch = preprocess.get_batch(train_labels, i*model.batch_size, model.batch_size)

        # indices = tf.range(start=0, limit=len(feature_inputs))
        # shuffled = tf.random.shuffle(indices)
        # numerical_input_batch = tf.gather(np.array(numerical_input_batch), shuffled)
        # feature_input_batch = tf.gather(np.array(feature_input_batch), shuffled)
        # label_batch = tf.gather(np.array(label_batch), shuffled)

        with tf.GradientTape() as tape:
            logits = model.call(numerical_input_batch, tf.convert_to_tensor(feature_input_batch), char_input_batch, is_training=None)
            loss = model.loss(logits, label_batch)
            losses.append(loss)
            print("Batch loss: {}".format(loss))
            print("Batch accuracy: {}".format(model.accuracy(logits, label_batch)))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return losses

def test(model, numerical_inputs, feature_inputs, char_inputs, test_labels):
    max_itr = len(feature_inputs) // model.batch_size
    accuracies = []
    for i in range(max_itr):
        numerical_input_batch = preprocess.get_batch(numerical_inputs, i*model.batch_size, model.batch_size)
        feature_input_batch = preprocess.get_batch(feature_inputs, i*model.batch_size, model.batch_size)
        char_input_batch = preprocess.get_batch(char_inputs, i*model.batch_size, model.batch_size)
        label_batch = preprocess.get_batch(test_labels, i*model.batch_size, model.batch_size)
        logits = model.call(numerical_input_batch, tf.convert_to_tensor(feature_input_batch), char_inputs, is_training=False)
        accuracies.append(model.accuracy(logits, label_batch))
    print("Testing accuracy: ", tf.reduce_mean(accuracies))

def visualize_loss(losses):
    """
    Uses Matplotlib to visualize loss per batch. Call this in train().
    When you observe the plot that's displayed, think about:
    1. What does the plot demonstrate or show?
    2. How long does your model need to train to reach roughly its best accuracy so far,
    and how do you know that?
    param losses: an array of loss value from each batch of train

    :return: doesn't return anything, a plot should pop-up
    """
    x = np.arange(1, len(losses)+1)
    plt.xlabel('i\'th Batch')
    plt.ylabel('Loss Value')
    plt.title('Loss per Batch')
    plt.plot(x, losses)
    plt.show()

def main():

    print("Preprocessing...")
    train_inputs, train_labels, test_inputs, test_labels = preprocess.get_data("../data/fma_metadata/tracks.csv")

    char_inputs = preprocess.make_char_dict(train_inputs)

    model = Model(len(char_inputs))

    numerical_train, numerical_test = preprocess.make_numerical_lists(train_inputs, test_inputs)
    feature_train, feature_test = preprocess.make_feature_lists(train_inputs, test_inputs)

    print("Training...")
    losses = []
    for epoch in range(1):
        losses.extend(train(model, numerical_train, feature_train, char_inputs, train_labels))

    print("Testing...")
    test(model, numerical_test, feature_test, char_inputs, test_labels)

    visualize_loss(losses)

if __name__ == '__main__':
    main()
