import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class Model(tf.keras.Model):
    def __init__(self):
        """
        The Model class predicts the genre of a track.
        """
        super(Model, self).__init__()

        self.batch_size = 200
        self.hidden_size = 264
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.dense_1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')

    def call(self, inputs, is_training=None):
        """
        :param inputs: shape [batch_size, features]
        """
        outputs = self.dense_1(inputs)
        outputs = self.dense_2(outputs)
        
        return outputs

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
