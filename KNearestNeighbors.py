def manhattan_distance(vec1, vec2):
    """Compute the manhattan distance between two vectors

    Parameters
    ----------
    vec1 : {array-like, sparse matrix},
    shape = [n_features]
        vector 1
    
    vec2 : {array-like, sparse matrix},
    shape = [n_features]
        vector 2
        
    Returns
    ----------
    dist : float
        Manhattan distance
    """
    dist = abs(vec1-vec2).sum()
    return dist

class NearestNeighbors(object):
    """Nearest Neighbors Classifier

    Parameters
    ----------        
    x_train : {array-like, sparse matrix},
    shape = [n_samples, n_features]
        Training data

    y_train : array-like, shape = [n_samples]
        Training targets

    k_limit : integer
        Maximum number of neighbors
        
    distance_functions : callable
        distance function to be used
    """
    def __init__(self, x_train, y_train, k_limit, distance_function):
        self.k_limit = k_limit
        self.distance_function = distance_function
        self.data = x_train
        self.labels = y_train
        
    def get_neighbors_single(self, input_vec):
        """Find the k limit # nearest neighbors to the input vector

        Parameters
        ----------
        input_vec : {array-like, sparse matrix},
        shape = [n_features]
            Input Vector

        Returns
        ----------
        neighbors : {array-like}
            array of nearest neighbor points to the input vector
        """
        distances = pd.DataFrame(np.apply_along_axis(self.distance_function, 1, self.data, input_vec))   
        nearest = distances.sort_values(0)[:self.k_limit]
        neighbors = np.ravel(self.labels.loc[nearest.index])
        return neighbors

    def predict_single(self, neighbors, k_val):
        """Make a single prediction

        Parameters
        ----------
        neighbors : {array-like, sparse matrix},
        shape = [n_samples]
            All neighbors
        
        k_val : Integer
            Number of nearest neighbors to consider
            
        Returns
        ----------
        class : Integer
            Predicted Class
        """
        class1 = np.sum(neighbors[:k_val])
        class0 = abs(class1 - k_val)
        if class1 > class0: return 1
        elif class1 < class0: return 0
        else: return int(random())

    def get_neighbors(self, test_set):
        """Find all neighbors for every point in the test set

        Parameters
        ----------
        test_set : {array-like, sparse matrix},
        shape = [n_samples, n_features]
            test data
        """
        self.all_neighbors = np.apply_along_axis(self.get_neighbors_single, 1, test_set)

    def predict(self, test_set):
        """Make predictions on the entire test set

        Parameters
        ----------
        test_set : {array-like, sparse matrix},
        shape = [n_samples, n_features]
            test data
        """
        self.get_neighbors(test_set)
        self.all_predictions = []
        for k in range(1, self.k_limit+1):
            self.all_predictions.append(pd.DataFrame(np.apply_along_axis(self.predict_single, 1, pd.DataFrame(self.all_neighbors), k)))
                
    def get_accuracy(self, test_labels):
        """Compute the accuracy for each k value 

        Parameters
        ----------
        test_labels : {array-like, sparse matrix},
        shape = [n_samples]
            test labels
            
        Returns
        ----------
        accuracies : List[Floats]
            Accuracies for each k value
        """
        accuracies = [((self.all_predictions[i] == test_labels).sum())/len(test_labels) for i in range(self.k_limit)]
        return accuracies