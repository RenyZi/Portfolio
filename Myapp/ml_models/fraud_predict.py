import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FraudPrediction:
    def __init__(self, samples = 1000, iterations = 1000, learningrate = 0.1, k =5, lambda_ = 1, regularization = None):
        
        np.random.seed(42) # for reproductivity

        self.k = k
        self.lambda_ = lambda_
        self.samples = samples
        self.iterations = iterations
        self.learningrate = learningrate
        self.regularization = regularization

        # variables
        self.theta = None
        self.Cost_history = None
        self.mean = None
        self.std = None
        self.fraud_data = None
        self.x = None
        self.y = None

        # functions for data generation and preprocessing
        self.dataGenerate()
        self.preprocess()


    # the dataset
    def dataGenerate(self):
        Amount = np.random.randint(50, 15001, size = self.samples)
        available = np.array(['Nairobi', 'Nakuru', 'Kisumu', 'Mombasa', 'Kakamega'])
        Location = np.random.choice(available, size = self.samples)
        Time_of_day = np.random.choice(['Morning','Afternoon','Evening','Night'], size = self.samples)
        Payment_method = np.random.choice(['Credit Card','Debit Card','Paypal','Bitcoin'], size = self.samples)
        Account_age = np.random.randint(1, 71, size = self.samples)

        # mapping the data data
        Location_mapping = {
            "Nairobi" : 1,
            "Nakuru" : 2,
            "Kisumu" : 3,
            "Mombasa" : 4,
            "Kakamega" : 5
        }

        Time_of_day_mapping = {
            "Morning" : 1,
            "Afternoon" : 2,
            "Evening" : 3,
            "Night" : 4
        }

        Payment_method_mapping = {
            "Credit Card" : 1,
            "Debit Card" : 2,
            "Paypal" : 3,
            "Bitcoin" : 4

        }

        # generating the encoded dataset
        Location_Encoded = np.array([Location_mapping[loc] for loc in Location])
        Time_of_day_Encoded = np.array([Time_of_day_mapping[time] for time in Time_of_day])
        Payment_method_Encoded = np.array([Payment_method_mapping[method] for method in Payment_method])

        # fraud 
        amount_normal = (Amount - 50) / (30000 - 50)
        normal_time = Time_of_day_Encoded / 23
        normal_account_age = (Account_age - 1) / (70 - 1)

        fraud = (0.5 + (0.6 * amount_normal) + (0.3 * normal_time) - (0.5 * normal_account_age)).astype(int)
        
        # sample dataset
        dataset = {
            "Amount(Ksh)" : Amount,
            "Location" : Location,
            "Time_of_day" : Time_of_day,
            "Payment_method" : Payment_method,
            "Account_age(Months)" : Account_age
        }

        self.fraud_data = pd.DataFrame.from_dict(dataset)
        self.fraud_data["Fraud"] = fraud
        self.fraud_data["Fraud"] = self.fraud_data["Fraud"].replace({1: "Fraud",0: "Not Fraud"})
        self.fraud_data = self.fraud_data.head(9)
        self.fraud_data = self.fraud_data.to_html(classes="table text-light", index=False, border=0)
        fraud_features = np.c_[Amount, Location_Encoded, Time_of_day_Encoded, Payment_method_Encoded, Account_age]
        
        self.x = fraud_features
        self.y = fraud

        return self.fraud_data

    # scaling the dataset
    def scaling(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        eps = 1e-8
        std = np.where(std == 0, eps, std)
        scaled = (X - mean) / std
        return mean, std, scaled
    
    # preprocessing
    def preprocess(self):
        self.mean, self.std, scaled_data = self.scaling(self.x)
        self.x_scaled = np.c_[np.ones(scaled_data.shape[0]), scaled_data]

    # sigmoid
    def sigmoid(self, z):
        result = 1 / (1 + np.exp(-z))
        return result
    # cost function
    def Loggloss(self, x, y, theta, lambda_=0, regularization=None):
        m = len(y)
        hypothisis = self.sigmoid(x @ theta)
        epsilion = 1e-8
        crossentropy = -(1/m) * np.sum( (y * np.log(hypothisis + epsilion)) + ((1 - y) * np.log(1 - hypothisis + epsilion)) )
        
        if regularization == 'ridge':
            crossentropy += (lambda_/m) * np.sum(theta[1:]**2)
        elif regularization == 'lasso':
            crossentropy += (lambda_/m) * np.sum(np.abs(theta[1:]))
        
        return crossentropy
    
    # gradient
    def GradientDescent(self, x, y, theta, iterations, learningrate, lambda_=0, regularization=None):
        m = len(y)
        costhistory = []

        for _ in range(iterations):
            hypothisis = self.sigmoid(x @ theta)
            diff = hypothisis - y
            gradient = (1/m) * x.T.dot(diff)

            if regularization == 'ridge':
                gradient[1:] += (lambda_ / m) * theta[1:]
            elif regularization == 'lasso':
                gradient[1:] += (lambda_ / m) * np.sign(theta[1:])

            theta -= learningrate * gradient
            costhistory.append(self.Loggloss(x, y, theta, lambda_, regularization))

        return theta, costhistory
    
    # cross validation
    def cross_validation(self, regularization=None):
        m = len(self.y)
        indeces = np.arange(0, m)
        np.random.shuffle(indeces)

        folds = np.array_split(indeces, self.k)
        Errors = []

        for fold in folds:
            test_idx = fold
            train_idx = np.setdiff1d(indeces, test_idx)

            # train dataset
            x_train, y_train = self.x_scaled[train_idx], self.y[train_idx]

            # test dataset
            x_test, y_test = self.x_scaled[test_idx], self.y[test_idx]

            theta = np.zeros(x_train.shape[1])
            theta,_ = self.GradientDescent(x_train, y_train, theta, self.iterations, self.learningrate, self.lambda_, regularization)

            prediction = self.sigmoid(x_test @ theta)
            logglos = -(1/m) * np.sum( y_test * np.log(prediction + 1e-8) + ((1-y_test) * np.log(1 - prediction + 1e-8)))
            Errors.append(logglos)

        return np.mean(Errors)
    
    # train 
    def train(self, regularization = None):
        theta = np.zeros(self.x_scaled.shape[1])
        self.theta, self.Cost_history = self.GradientDescent(
            self.x_scaled, self.y, theta, self.iterations, self.learningrate, self.lambda_, regularization
            )
        
        return self.theta
    
    # prediction
    def prediction(self, new_data):
        new_data = new_data.reshape(1, -1)
        new_data_scaled = (new_data - self.mean) / self.std
        new_data_normalized = np.c_[np.ones((new_data_scaled.shape[0], 1)), new_data_scaled]

        predict = self.sigmoid(new_data_normalized @ self.theta)

        if predict >= 0.5:
            return "The transaction is fraudulent!"
        else:
            return "The transaction is not a fraud"


    # plot
    def plot(self):
        if self.Cost_history:
            plt.figure(figsize=(8,6))
            plt.title("Fraudulent gradient descent converging")
            plt.plot(range(self.iterations), self.Cost_history)
            plt.ylabel("Cross Entropy")
            plt.xlabel("Iterations")
            plt.show()

