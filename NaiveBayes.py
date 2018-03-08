def get_bernoulli_prob(x, param):
    """Bernoulli probability density function

    Parameters
    ----------
    x : {array-like, sparse matrix},
    shape = [n_samples, n_features]
        data
        
    param : float
        Probability of "success"
        
    Returns
    ----------
    bernoulli_prob : float
        Bernoulli probability
    """  
    bernoulli_prob = (param**x) * (1-param)**(1-x)
    return bernoulli_prob

def get_pareto_prob(x, param):
    """Pareto probability density function

    Parameters
    ----------
    x : {array-like, sparse matrix},
    shape = [n_samples, n_features]
        data
        
    param : float
        Pareto distribution parameter
        
    Returns
    ----------
    pareto_prob : float
        Pareto probability
    """  
    pareto_prob = param * (x**(-1*(param + 1)))
    return pareto_prob

class NaiveBayes(object):
    """Naive Bayes Classifier

    Parameters
    ----------        
    x_train : {array-like, sparse matrix},
    shape = [n_samples, n_features]
        Training data

    y_train : array-like, shape = [n_samples]
        Training targets

    num_classes : integer
        Number of classes
    """
    def __init__(self, x_train, y_train, num_classes):
        self.values_per_class = [x_train[y_train[0] == i] for i in range(num_classes)]
        self.amount_per_class = [len(self.values_per_class[i]) for i in range(len(self.values_per_class))]
        self.total = len(y_train)
        
    def find_parameters(self):
        """Find the probability parameters for each class
        
        """
        self.probability_params = {}
        for i in range(len(self.values_per_class)):
            params = self.values_per_class[i].mean()
            pareto_num = len(self.values_per_class[i])
            pareto_denom = (np.log(self.values_per_class[i][list(self.values_per_class[i].columns[54:57])]).sum())
            params.update(pareto_num/pareto_denom)
            self.probability_params[i] = params

    def get_probabilities(self, input_vec):
        """Calculate the probabilities of classes using the features of the input vector
        
        input_vec : {array-like, sparse matrix},
        shape = [1, n_features]
            Input vector
            
        Returns
        ----------
        probabilities : list[float]
            Probabilities per class of the input vector
        """
        probabilities = []
        class_num = 0
        for class_val, param in self.probability_params.items():
            probability = self.amount_per_class[class_num]/self.total # CHANGE TO USE CLASS PROBABILITY
            for i in range(len(param)):
                if i < 54:
                    probability *= get_bernoulli_prob(input_vec[i], param[i])
                else:
                    probability *= get_pareto_prob(input_vec[i], param[i])
            probabilities.append(probability)
            class_num += 1
        return probabilities

    def predict_single(self, input_vec):
        """Make a single prediction
        
        input_vec : {array-like, sparse matrix},
        shape = [1, n_features]
            Input vector
            
        Returns
        ----------
        prediction : integer
            Predicted class
        """
        probabilities = self.get_probabilities(input_vec)
        prediction = probabilities.index(max(probabilities))
        return prediction

    def predict(self, test_set):
        """Make predictions on the entire test set
        
        test_set : {array-like, sparse matrix},
        shape = [n_samples, n_features]
            Test set to make predictions on
            
        Returns
        ----------
        predictions : List[integer]
            List of predicted classes for each point in the testset
        """
        predictions = [self.predict_single(feature_vec[1]) for feature_vec in test_set.iterrows()]
        return predictions

    def accuracy(self, predictions, y_test):
        """Get the accuracy and errors
        
        predictions : List[integer],
            List of predicted classes
            
        y_test : array-like, shape = [n_samples]
            Actual classes
            
        Returns
        ----------
        (Integer, confusion_matrix)
            Accuracy, confusion matrix
        """
        prediction_matrix = confusion_matrix(y_test, predictions, labels = [0, 1])
        accuracy = np.diagonal(prediction_matrix).sum()/len(predictions)*100
        return (accuracy, prediction_matrix)
        