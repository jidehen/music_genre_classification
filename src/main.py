import preprocess
from load_audio import walk_files, read_from_npy
import numpy as np
import tensorflow as tf
import rnn
import dense
import matplotlib.pyplot as plt

def train(model, train_inputs, train_labels):

    indices = tf.range(start=0, limit=len(train_inputs))
    shuffled = tf.random.shuffle(indices)
    train_inputs = tf.gather(train_inputs, shuffled)
    train_labels = tf.gather(train_labels, shuffled)

    num_batches = len(train_inputs) // model.batch_size
    losses = []

    for batch in range(num_batches):
        start = batch * model.batch_size
        end = (batch + 1) * model.batch_size

        print("Batch " + str(batch) + " / " + str(num_batches))

        input_batch = train_inputs[start:end]
        label_batch = train_labels[start:end]

        with tf.GradientTape() as tape:
            probs = model.call(input_batch)
            loss = model.loss(probs, label_batch)
            losses.append(loss)
            print("Batch loss: {}".format(loss))
            print("Batch accuracy: {}".format(model.accuracy(probs, label_batch)))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return losses

def test(model, test_inputs, test_labels):
    num_batches = len(test_inputs) // model.batch_size

    loss = 0
    accuracies = []

    for batch in range(num_batches):
        start = batch * model.batch_size
        end = (batch + 1) * model.batch_size

        input_batch = test_inputs[start:end]
        label_batch = test_labels[start:end]

        probs = model.call(input_batch)

        loss += model.loss(probs, label_batch)
        accuracies.append(model.accuracy(probs, label_batch))

    print("Testing accuracy: ", tf.reduce_mean(accuracies))

def visualize_loss(losses):
    """
    Uses Matplotlib to visualize loss per batch. Call this in train().
    When you observe the plot that's displayed, think about:
    1. What does the plot demonstrate or show?
    2. How long does your model need to train to reach roughly its best accuracy so far,
    and how do you know that?
    Optionally, add your answers to README!
    param losses: an array of loss value from each batch of train

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    x = np.arange(1, len(losses)+1)
    plt.xlabel('i\'th Batch')
    plt.ylabel('Loss Value')
    plt.title('Loss per Batch')
    plt.plot(x, losses)
    plt.show()

def main():
    # model = rnn.Model()
    np.set_printoptions(threshold=np.inf)
    num_model = dense.Model()
    print("Preprocessing...")
    train_inputs, train_labels, test_inputs, test_labels = preprocess.get_data("../data/fma_metadata/tracks.csv")
    print("Preprocessing Complete")

    train_listens = np.asarray([x.listens for x in train_inputs])
    train_interest = np.asarray([y.comments for y in train_inputs])
    train_favorites = np.asarray([z.favorites for z in train_inputs])
    train_inputs = np.stack((train_listens, train_interest, train_favorites), axis=1)

    test_listens = np.asarray([x.listens for x in test_inputs])
    test_interest = np.asarray([y.comments for y in test_inputs])
    test_favorites = np.asarray([z.favorites for z in test_inputs])
    test_inputs = np.stack((test_listens, test_interest, test_favorites), axis=1)

    print("Training...")
    losses = []
    for epoch in range(1):
        # losses = train(model, train_inputs=train_feat_inputs, train_labels=train_labels)
        losses = train(num_model, train_inputs=train_inputs, train_labels=train_labels)
    print("Training Complete")
    print("Testing...")
    # test(model, test_inputs=test_feat_inputs, test_labels=test_labels)
    test(num_model, test_inputs=test_inputs, test_labels=test_labels)
    visualize_loss(losses)

if __name__ == '__main__':
    main()
