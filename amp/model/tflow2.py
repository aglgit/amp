import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from . import LossFunction
from ..utilities import ConvergenceOccured


class NeuralNetwork(Model):
    def __init__(self,
                 loss_function=None,
                 hidden_layers=(10, 10),
                 activation="tanh",
                 keep_prob=1.0,
                 batch_size=1,
                 learning_rate=4e-3,
                 regularization=None,
                 optimizer="ADAM",
                ):
        super(NeuralNetwork, self).__init__()

        if loss_function is None:
            convergence = {"energy_rmse": 1e-3,
                           "force_rmse": None, 
                           "max_steps": int(1e3), }
            energy_coefficient=1.0
            force_coefficient=None
            loss_function = LossFunction(
                                        convergence=convergence,
                                        energy_coefficient=energy_coefficient,
                                        force_coefficient=force_coefficient)
        self.loss_function = loss_function

        self.optimizer = tf.keras.optimizers.get(optimizer)
        if regularization is not None:
            self.regularizer = tf.keras.regularizers.l2(regularization)

        self.batch_size = batch_size

        self.importname = ".model.neuralmetwork.tflow2"

        self.hidden_layers = []
        for nodes in hidden_layers:
            self.hidden_layers.append(Dense(nodes, activation=activation))
        self.hidden_layers.append(Dense(1, activation="linear"))

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)

        return x

    def derivative(self, x):
        with tf.GradientTape() as t:
            t.watch(x)
            y = self.call(x)

        return t.gradient(y, x)

    def fit(self, trainingimages, descriptor, parallel, log=None):
        self.log = log
        lf = self.loss_function

        images = trainingimages
        keys = images.keys()
        fingerprintDB = descriptor.fingerprints
        if lf.force_coefficient is None:
            fingerprintprimeDB = None
        else:
            fingerprintprimeDB = descriptor.fingerprintprimes

        for key in keys:
            fingerprint
