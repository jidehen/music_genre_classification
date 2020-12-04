import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data, get_batch

class Model(tf.keras.Model):
    def __init__(self):
        """
        The Model class predicts the genre of a track.
        """
        super(Model, self).__init__()

        self.batch_size = 400
        self.hidden_size = 100
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.num_classes = 16

        self.dense_layer = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.softmax_layer = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, inputs):
        """
        :param inputs: shape [batch_size, features]
        """

        return self.softmax_layer(self.dense_layer(inputs))

    def loss(self, probs, labels):
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, probs))

    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number
        of correct predictions with the correct answers.
        :param probabilities: result of running model.call() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy
        count = 0
        for x in range(len(probabilities)):
             if np.argmax(labels[x]) == np.argmax(probabilities[x]):
                count += 1
        return count/len(probabilities)
