import numpy as np
import random
from numpy.linalg import norm

class Perceptron:
    def __init__(self, perceptrons, learning_rate=0.01, threshold=0.9):
        """ Initializes the perceptron """
        self.learning_rate = learning_rate  # Learning rate
        self.threshold = threshold  # Threshold

        self.perceptrons = perceptrons  # List of perceptrons
        self.perceptrons_count = len(perceptrons)  # Number of perceptrons
        
        # Use for learning
        self.feedback = np.zeros(self.perceptrons_count)
        self.theta = np.zeros(self.perceptrons_count + 1)
        self.action = 0
        self.do = False

        # Use for counting feedback loops
        self.trace = 0
        self.simi = 1
        self.loop_count = 0
        self.feedback_cache = self.feedback.copy()

    def reset(self):
        """ Resets the state of the perceptron """
        self.feedback = np.zeros(self.perceptrons_count)
        self.feedback_cache = self.feedback.copy()
        self.action = 0
        self.do = False
        self.simi = 1
        self.loop_count = 0

    # def activation(self):
    #     z = np.sum(self.theta * np.append(self.feedback, 1))
    #     return max(0.01 * z, z)

    def activation(self):
        """ Calculates the activation function's value """
        z = np.sum(self.theta * np.append(self.feedback, 1))
        return 1 / (1 + np.exp(-z))

    def cal(self, disable=False):
        """ Calculates action value """
        self.action = self.activation()
        if disable:
            self.do = False
            return
        if random.random() < self.learning_rate:
            self.do = True
            self.action = random.random()
        else:
            self.do = False

    def similarity(self, a1, a2):
        """ Calculates cosine similarity between two vectors """
        if norm(a1) > 0 and norm(a2) > 0:
            return np.dot(a1, a2) / (norm(a1) * norm(a2))
        return 1

    def update(self):
        """ Updates the state of the perceptron """
        self.feedback = np.array([p.action for p in self.perceptrons])

        similarity = self.similarity(self.feedback_cache, self.feedback)
        if (self.simi - self.threshold) * (similarity - self.threshold) < 0:
            self.loop_count += 0.5
        self.simi = similarity

        if self.do:
            action = self.activation()
            grad = (action - self.action) * np.append(self.feedback, 1)

            # Updates tracking value
            self.trace = max(self.trace - 1, int(self.loop_count))

            # Decides whether to update weights
            if random.random() < 1.0 / np.exp(self.trace):
                self.theta -= self.learning_rate * grad

            self.loop_count = 0
            self.feedback_cache = self.feedback.copy()
