import preprocess
from load_audio import walk_files, read_from_npy
import numpy as np
import tensorflow as tf
from rnn import Model
import matplotlib.pyplot as plt

def train(model, train_inputs, train_labels):
    max_itr = len(train_inputs) // model.batch_size
    losses = []
    for i in range(max_itr):
        print("Batch " + str(i+1) + " / " + str(max_itr))
        input_batch = preprocess.get_batch(train_inputs, i*model.batch_size, model.batch_size)
        label_batch = preprocess.get_batch(train_labels, i*model.batch_size, model.batch_size)
        indices = tf.range(start=0, limit=len(input_batch))
        shuffled = tf.random.shuffle(indices)
        input_batch = tf.gather(np.array(input_batch), shuffled)
        label_batch = tf.gather(np.array(label_batch), shuffled)
        with tf.GradientTape() as tape:
            logits = model.call(input_batch, is_training=None)
            loss = model.loss(logits, label_batch)
            losses.append(loss)
            print("Batch loss: {}".format(loss))
            print("Batch accuracy: {}".format(model.accuracy(logits, label_batch)))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return losses

def test(model, test_inputs, test_labels):
    max_itr = len(test_inputs) // model.batch_size
    accuracies = []
    for i in range(max_itr):
        input_batch = preprocess.get_batch(test_inputs, i*model.batch_size, model.batch_size)
        label_batch = preprocess.get_batch(test_labels, i*model.batch_size, model.batch_size)
        logits = model.call(input_batch, is_training=False)
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
    model = Model()
    print("Preprocessing...")
    train_inputs, train_labels, test_inputs, test_labels = preprocess.get_data("../data/fma_metadata/tracks.csv")
    print(tf.shape(train_inputs))
    print("Training...")
    losses = []
    for epoch in range(3):
        print("Epoch {}".format(epoch))
        loss = train(model, train_inputs=train_inputs, train_labels=train_labels)
        losses.extend(loss)

    print("Testing...")
    test(model, test_inputs=test_inputs, test_labels=test_labels)
    visualize_loss(losses)
    
if __name__ == '__main__':
    main()
