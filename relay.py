"""RELAY: Rule Extraction from LAbeled observations for Your classification"""

import numpy as np


class RELAY:
    """RELAY: An experimental algorithm for training a machine learning model through rule-based methods and weighing the importance of features, along with verbal translations for the user to understand the model's predictions"""

    def __init__(self, training_data: np.ndarray[np.ndarray], shape: tuple):
        """Experimental algorithm for training a machine learning model

        Args:
            training_data (np.ndarray[np.ndarray]): All of the trainig data that will be used to train the model. The labels are stored in the actual index. Label 1 is stored in index 1, label 2 is stored in index 2, and so on. The subarrays are the features that will be used to train the model. The shape of the training data must me provided.

            shape (tuple): The shape of the training data. The first index is the number of features, and the second index is the number of samples. EACH OF THE SUBARRAYS IN THE TRAINING DATA MUST HAVE THE SAME LENGTH!
        """
        self.training_data = training_data
        self.shape = shape
        self.gen_weights()

    def gen_weights(self):
        """Generate weights for each feature in every subarray of the training data by getting the average of all feature all subarrays. Then, make a weights array for each label in the training data contianing the weights for each feature in the subarrays. The weights will be used to determine the importance of each feature in the training data. The weights will be calculated based on how far the feature is from the average of all features in the training data."""
        self.weights = np.zeros((self.shape[0], self.shape[2]))
        self.avg = np.zeros((self.shape[0], self.shape[2]))
        for i in range(self.shape[0]):
            self.avg[i] = np.mean(self.training_data[i], axis=0)
        self.global_avg = np.mean(self.avg, axis=0)
        for i in range(self.shape[0]):
            for j in range(self.shape[2]):
                self.weights[i][j] = abs(self.global_avg[j] - self.avg[i][j])

    def train(self):
        """Train the model with the training data. The model will be trained with the training data that was provided in the constructor using the following steps:
        1. Generate weights
        """
        self.gen_weights()


if __name__ == "__main__":
    arr1 = np.array([[1, 2, 3], [4, 5, 6]])
    arr2 = np.array([[7, 8, 9], [10, 11, 12]])
    arr3 = np.array([[13, 14, 15], [16, 17, 18]])
    arrays = np.stack((arr1, arr2, arr3), axis=0)
    model = RELAY(arrays, arrays.shape)
    # model.train()
