import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AttritionPredict:
    def __init__(self, samples = 1000, epoch = 1000, alpha = 0.1, k = 5, lambda_ = 1, regularization = None):

        np.random.seed(42) # for reproductivity

        self.samples = samples
        self.epoch = epoch
        self.alpha = alpha
        self.k = k
        self.lambda_ = lambda_
        self.regularization = regularization

        # other variables
        self.theta = None
        self.Costhistory = None
        self.mean = None
        self.std = None
        self.attrition = None
        self.x = None
        self.y = None

        # helper function
        self.generator()
        self.preprocess()

    # dataset
    def generator(self):
        Age = np.random.randint(22, 71, size = self.samples)
        JobLevel = np.random.randint(1,6, size = self.samples)
        MonthlyIncome = np.random.randint(500, 1001, size = self.samples) * 100
        Workload = np.random.randint(40, 61, size = self.samples)
        Jobsatisfaction = np.random.randint(1,6, size = self.samples)
        Commute_distance = np.random.randint(8, 31, size = self.samples)

        attr = (((Age >= 55) & (JobLevel < 2) & ((Jobsatisfaction <= 2))) | ((MonthlyIncome < 55000) & (Workload >= 55) & (Commute_distance > 15))).astype(int)

        data = {
            "Age":Age,
            "Job Level":JobLevel,
            "Monthly Income":MonthlyIncome,
            "Workload(Hrs/week)":Workload,
            "Jobsatisfaction":Jobsatisfaction,
            "Commute_distance":Commute_distance
        }

        self.attrition = pd.DataFrame.from_dict(data)
        self.attrition["Attrition"] = attr
        self.attrition["Attrition"] = self.attrition["Attrition"].replace({0:"No", 1:"Yes"})
        self.attrition = self.attrition.head(9)
        self.attrition = self.attrition.to_html(classes="table text-light", index=False, border=0)

        features = np.c_[Age, JobLevel, MonthlyIncome, Workload, Jobsatisfaction, Commute_distance]
        self.x = features
        self.y = attr

        return self.attrition
        

    # scaling
    def scaling(self, feature):
        mean = np.mean(feature, axis=0)
        std = np.std(feature, axis=0)
        std = np.where(std == 0, 1e-6, std)
        scaled = (feature - mean) / std
        return mean, std, scaled
    
    # preprocessing
    def preprocess(self):
        self.mean, self.std, Scaled = self.scaling(self.x)
        self.x_scaled = np.c_[np.ones(Scaled.shape[0]), Scaled]


    # sigmoid 
    def sigmoid(self,z):
        g = 1 / (1 + np.exp(-z))
        return g
    
    # crossentropy
    def CrossEntropy(self, x, y, theta, lambda_=0, regularization=None):
        m = len(y)
        hypothesis = self.sigmoid(x.dot(theta))
        epsilion = 1e-8
        logloss = -(1/m) * np.sum( y * np.log(hypothesis + epsilion) + ((1-y) * np.log(1 - hypothesis + epsilion)))

        if regularization == 'ridge':
            logloss += (lambda_/m) * np.sum(theta[1:]**2)
        elif regularization == 'lasso':
            logloss += (lambda_/m) * np.sum(np.abs(theta[1:]))

        return logloss
    
    # gradient
    def GradienrDescent(self, x, y, theta, alpha, epoch, lambda_=0, regularization=None):
        m = len(y)
        costhistory = []

        for _ in range(epoch):
            hypothesis = self.sigmoid(x.dot(theta))
            diff = hypothesis - y
            gradient = (1/m) * x.T.dot(diff)

            if regularization == 'ridge':
                gradient[1:] += (lambda_/m) * theta[1:]
            elif regularization == 'lasso':
                gradient[1:] += (lambda_/m) * np.sign(theta[1:])

            theta -= gradient * alpha
            costhistory.append(self.CrossEntropy(x, y, theta, lambda_, regularization))

        return theta, costhistory
    
    # cross validation
    def KFold(self, regularization=None):
        m = len(self.y)
        indices = np.arange(0, m)
        np.random.shuffle(indices)
        folds = np.array_split(indices, self.k)
        Errors = []

        for fold in folds:
            test_index = fold
            train_index = np.setdiff1d(indices, test_index) # indices that are not in test index

            # train dataset
            x_train, y_train = self.x_scaled[train_index], self.y[train_index]

            # test dataset
            x_test, y_test = self.x_scaled[test_index], self.y[test_index]

            theta = np.zeros(x_train.shape[1])
            theta,_ = self.GradienrDescent( x_train, y_train, theta, self.alpha, self.epoch, self.lambda_, regularization)

            # test model
            prediction = self.sigmoid(x_test.dot(theta))
            eps = 1e-8
            loglos = -(1/m) * np.sum(y_test * np.log(prediction + eps) + ((1-y_test) * np.log(1 - prediction + eps)))
            Errors.append(loglos)

        return np.mean(Errors)
    
    # train
    def train(self, regularization=None):
        theta = np.zeros(self.x_scaled.shape[1])
        self.theta, self.Costhistory = self.GradienrDescent(self.x_scaled, self.y, theta, self.alpha, self.epoch, self.lambda_, regularization)
        return self.theta
    
    # ploting
    def plot(self):
        if self.Costhistory:
            plt.figure(figsize=(8,6))
            plt.plot(range(self.epoch), self.Costhistory)
            plt.title("Cost function converging")
            plt.ylabel("Cost function")
            plt.xlabel("Iterations")
            plt.show()

    # prediction
    def predict(self, new_attrition_data):
        new_attrition_data = new_attrition_data.reshape(1,-1)
        new_attrition_data_scaled = (new_attrition_data - self.mean) / self.std
        new_attrition_data_normalized = np.c_[np.ones(new_attrition_data_scaled.shape[0]), new_attrition_data_scaled]
        prediction = self.sigmoid(new_attrition_data_normalized @ self.theta)

        if prediction >= 0.01:
            return "The employee is likely to leave the company! "
        else:
            return "The employee is predicted not to leave the company "




