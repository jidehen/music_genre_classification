import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

from attenvis import AttentionVis

av = AttentionVis()

class Transformer_Model(tf.keras.Model):
	def __init__(self):

		super(Transformer_Model, self).__init__()

		# TODO:
		# 1) Define any hyperparameters
		# 2) Define embeddings, encoder, decoder, and feed forward layers
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

		# Define batch size and optimizer/learning rate
		self.batch_size = 350
		
		self.feature_size = 14

		# Define encoder and decoder layers:
		self.encoder = transformer.Transformer_Block(self.feature_size, False, False)
		#self.decoder = transformer.Transformer_Block(self.embedding_size, True, False)
		self.num_classes = 8

		self.dense_1 = tf.keras.layers.Dense(self.num_classes, activation='softmax')

		# Define dense layer(s)
		self.soft_max = tf.keras.layers.Dense(self.num_classes, activation='softmax')

	@tf.function
	def call(self, encoder_input, is_training=None):
		encoded = self.encoder.call(encoder_input)[:, :, 0]
		print(encoded[0][1])
		outputs = self.soft_max(encoded)
		#return outputs

	def accuracy(self, probabilities, labels):
		count = 0
		for x in range(np.shape(probabilities)[0]):
			if np.argmax(labels[x]) == np.argmax(probabilities[x]):
				count += 1
		return count/len(probabilities)


	def loss(self, prbs, labels):
		"""
		Calculates the model cross-entropy loss after one forward pass
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

		:param prbs:  float tensor, genre prediction probabilities [batch_size x num_classes]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""

		# Note: you can reuse this from rnn_model.
		return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, prbs))

	@av.call_func
	def __call__(self, *args, **kwargs):
		return super(Transformer_Model, self).__call__(*args, **kwargs)