import tensorflow as tf
import numpy as np

from model.pqca import PQCAModel

class DataFMNISTModel(PQCAModel):
    """
    Used to create a stilted quantum dataset from the classical fashion MNIST
    dataset using a specified CQCA ansatz based on a `PQCAModel` class.

    Args:
        qubits (cirq.Qubit): The logical qubits.
        num_layers_enc (int): The number of encoding layers.
        num_layers_var(int); The number of variational layers.
        cqca (list of list): List of list of strings of 'CZx', 'H', 'S', 
                                which make up the translationally invariant 
                                cqcas, where x is the distance of the CZ.
        observables (list of cirq.ops): The measured observables.
        train (bool): Whether the tfq layer is trainable or not.
        single_param (bool): Whether or not there is only one variational param
                             per layer. Default: False
    """
    def __init__(self, qubits, num_layers_enc, num_layers_var, 
                 cqca, observables, train, single_param=False):
        super().__init__(qubits, num_layers_enc, num_layers_var, 
                 cqca, observables, train)
        
        # initialize attributes as None
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test =  None, None 
        self.x_train_save, self.x_test_save = None, None
        self.std = None

    def create_fMNIST(self, num_train, num_test, norm):
        """
        Generates the pre-process fashion MNIST dataset.
        
        Args:
            num_train (int): Size of training data.
            num_test (int): Size of test and validation data.
            norm (bool): Whether the output is normalized.

        Returns:
            x_train (tf.Tensor): Training dataset input.
            y_train (tf.Tensor): Training dataset label.
            x_test (tf.Tensor): Validation dataset input.
            y_test (tf.Tensor): Validation dataset label.
            x_test2 (tf.Tensor): Test dataset input.
            y_test2 (tf.Tensor): Test dataset label.
            x_train_save (np.ndarray): Classical data input.
            x_test_save (np.ndarray): Classical validation data input.
            x_test2_save (np.ndarray): Classical validation data input. 
        """
        # Load raw dataset
        (x_train, y_train), \
            (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # PCA and component-wise normalization
        x_train, x_test = DataFMNISTModel._truncate_x(x_train, x_test, 
                                                n_components=self.num_qubits
                                                )
        x_mean, x_std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
        x_train, x_test = (x_train - x_mean) / x_std, (x_test - x_mean) / x_std

        # Prune dataset
        x_train = x_train[:num_train]
        x_test2 = x_test[num_test:2*num_test]
        x_test = x_test[:num_test]

        # save data for classical methods and compute the feature vectors of 
        # Havlivcek's encoding
        x_train_save = x_train.numpy()
        x_test_save = x_test.numpy()
        x_test2_save = x_test2.numpy()
        
        x_train = tf.convert_to_tensor(DataFMNISTModel._preprocess(
                                                    x_train.numpy(), 
                                                    self.num_qubits))
        x_test = tf.convert_to_tensor(DataFMNISTModel._preprocess(
                                                    x_test.numpy(), 
                                                    self.num_qubits))
        x_test2 = tf.convert_to_tensor(DataFMNISTModel._preprocess(
                                                    x_test2.numpy(), 
                                                    self.num_qubits))

        # Compute new labels and normalize
        y_train = self.model(x_train)
        y_test = self.model(x_test)
        y_test2 = self.model(x_test2)
        
        if norm:
            std = np.std(y_train)
            self.std = std
            y_train, y_test, y_test2 = y_train/std, y_test/std, y_test2/std

        self.x_train, self.y_train = x_train, y_train 
        self.x_test, self.y_test =  x_test, y_test
        self.x_test2, self.y_test2 = x_test2, y_test2
        self.x_train_save, self.x_test_save = x_train_save, x_test_save 
        self.x_test2_save  = x_test2_save

        return x_train, y_train, x_test, y_test, \
               x_test2, y_test2, x_train_save, x_test_save, x_test2_save

    @staticmethod
    def _truncate_x(x_train, x_test, n_components):
        """
        Perform PCA on image dataset, keeping the top n_components.
        
        Args:
            x_train (np.ndarray): Classical training dataset.
            x_test (np.ndarray): Classical testing dataset.
            n_components (int): Number of components to keep.
        
        Returns:
            x_train (tf.Tensor): Classical training dataset truncated by PCA.
            x_test (tf.Tensor): Classical testing dataset truncated by PCA.
        """
        n_points_train = tf.gather(tf.shape(x_train), 0)
        n_points_test = tf.gather(tf.shape(x_test), 0)

        # Flatten to 1D
        x_train = tf.reshape(x_train, [n_points_train, -1])
        x_test = tf.reshape(x_test, [n_points_test, -1])

        # Normalize
        feature_mean = tf.reduce_mean(x_train, axis=0)
        x_train_normalized = x_train - feature_mean
        x_test_normalized = x_test - feature_mean

        # Truncate
        e_values, e_vectors = tf.linalg.eigh(
            tf.einsum('ji,jk->ik', x_train_normalized, x_train_normalized))
        return tf.einsum('ij,jk->ik', x_train_normalized, 
                         e_vectors[:,-n_components:]
                         ), tf.einsum('ij,jk->ik', x_test_normalized, 
                                      e_vectors[:, -n_components:])

    @staticmethod
    def _preprocess(x, num_qubits):
        """
        Computes the feature vectors of Havlivcek's encoding.
        
        Args:
            x (np.ndarray): Input dataset.
            num_qubits (int): Number of qubits (which is also the datasize).

        Returns:
            x (np.ndarray): Input reformatted for Havlicek's encoding.
        """
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                x = np.append(x, np.transpose([x[:,i]*x[:,j]]), axis=1)
        return x
