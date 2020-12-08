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
        self.numerical_hidden_size = 100
        # self.feature_hidden_size = 164
        # self.hidden_size = 264
        # self.rnn_size = 256
        self.num_classes = 8
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.num_dense1 = tf.keras.layers.Dense(self.numerical_hidden_size, activation='relu')
        # self.feat_dense1 = tf.keras.layers.Dense(self.feature_hidden_size, activation='relu')
        #
        # self.LSTM = tf.keras.layers.GRU(self.rnn_size, return_sequences=False, return_state=True, dropout=.4, dtype=np.float64)
        #
        # self.dense_layer = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.softmax_layer = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    def call(self, numerical_inputs, feature_inputs, initial_state=None, is_training=True):
        """
        :param inputs: shape [batch_size, features]
        """

        numerical_output1 = self.num_dense1(numerical_inputs)
        numerical_output = self.softmax_layer(numerical_output1)
        # lstm_output, _ = self.LSTM(feature_inputs, initial_state=initial_state, training=is_training)
        # feature_output = self.feat_dense1(lstm_output)
        #
        # inputs = np.stack((numerical_output, feature_output), axis=1)
        #
        # output = self.softmax_layer(self.dense_layer(inputs))

        return numerical_output

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
