class LogsiticRegression(object):
    """Nearest Neighbors Classifier
    
    Parameters
    ----------        
    x_train : {array-like, sparse matrix},
    shape = [n_samples, n_features]
        Training data
    
    y_train : array-like, shape = [n_samples]
        Training targets
    """
    def __init__(self, x_train, y_train):
        x_train[len(y_train)] = 1
        self.x = x_train.as_matrix()
        self.y = pd.DataFrame(y_train[0].replace(0,-1,inplace=False)).as_matrix()
        self.xy = self.y * self.x
        
    def steepestAscent(self, iterations):
        """Compute weights using the steepest ascent algorithm
        
        Parameters
        ----------        
        iterations : Integer,
            Number of iterations to run
        """
        w = np.zeros(shape = (58,1))
        xyw = np.dot(self.xy, w)
        self.objectives = []
        for t in range(1, iterations+1):
            eta = (1/(10**5 * np.sqrt(t+1)))
            w += pd.DataFrame(eta*self.get_gradient(xyw))
            xyw = np.dot(self.xy, w)
            self.objectives.append(self.get_objective(xyw))
        self.w = w
        
    def newtonsMethod(self, iterations):
        """Compute weights using Newton's method
        
        Parameters
        ----------        
        iterations : Integer,
            Number of iterations to run
        """
        w = np.zeros(shape = (58,1))
        xyw = np.dot(self.xy, w)
        self.objectives = []
        for t in range(1, iterations+1):
            eta = (1/(np.sqrt(t+1)))
            gradient = self.get_gradient(xyw)
            
            xw = np.dot(self.x, w)
            
            hessian_inv = np.linalg.inv(pd.DataFrame(self.get_hessian(xw)))
            step = np.dot(hessian_inv, gradient)
            
            w -= pd.DataFrame(eta*step)
            xyw = np.dot(self.xy, w)

            self.objectives.append(self.get_objective(xyw))
        self.w = w
    
    def get_gradient(self, xyw):
        """Compute the gradient
        
        Parameters
        ----------        
        xyw : array-like
            
        Returns
        ---------- 
        grad: array-like
            gradient
        """
        sigs = 1 - scipy.special.expit(xyw)
        grad = np.sum(sigs*self.xy, axis = 0)
        return grad
    
    def get_hessian(self, xw):
        """Compute the Hessian matrix
        
        Parameters
        ----------        
        xw : array-like
            
        Returns
        ---------- 
        hess: array-like
            hessian
        """
        sigmoids = scipy.special.expit(xw)
        sig_product = sigmoids*(1-sigmoids)
        x_product = sig_product.T*self.x.T
        hess = -1*np.dot(x_product, self.x)
        return hess
        
    def get_objective(self, val):
        """Compute the Objective function
        
        Parameters
        ----------        
        val : array-like
            
        Returns
        ---------- 
        obj: float
            Value of objective function
        """
        obj = np.sum(np.ma.log(scipy.special.expit(val)))
        return obj
    
    def predict(self, test_set):
        """Make predictions on the entire test set
        
        test_set : {array-like, sparse matrix},
        shape = [n_samples, n_features]
            Test set to make predictions on
        """
        test_set[57]=1
        sigmoids = scipy.special.expit(np.dot(test_set, self.w))
        self.predictions = [1 if sigmoid >= 0.5 else -1 for sigmoid in sigmoids]

    def get_accuracy(self, test_labels):
        """Compute the accuracy

        Parameters
        ----------
        test_labels : {array-like, sparse matrix},
        shape = [n_samples]
            test labels
            
        Returns
        ----------
        accuracy : Float
            Model accuracy on test set
        """
        test_labels = pd.DataFrame(test_labels[0].replace(0,-1,inplace=False))
        return (self.predictions == test_labels).sum()/len(self.predictions)*100