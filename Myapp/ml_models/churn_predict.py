import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ChurnPrediction:
    def __init__(self, sample=1000, epoch=1000, alpha=0.1, k=5, lam_=0.01, regularization=None):
        np.random.seed(42)

        self.sample = sample
        self.epoch = epoch
        self.alpha = alpha
        self.k = k
        self.lam_ = lam_
        self.regularization = regularization

        # variables
        self.theta = None
        self.CostHistory = None
        self.churn_data = None
        self.mean = None
        self.std = None

        self.x = None
        self.y = None

        # load and process data
        self.dataset()
        self.process()

    def dataset(self):
        Age = np.random.randint(22, 65, size=self.sample)
        MonthlySpending = np.random.randint(20, 121, size=self.sample)
        Tenure = np.random.randint(6, 73, size=self.sample)
        ContractType = np.random.choice(['Monthly', 'Yearly'], size=self.sample)
        InternetUsage = np.random.randint(5, 31, size=self.sample)
        SupportCalls = np.random.randint(0, 11, size=self.sample)

        # Encode contract type
        ContractTypeMapping = {"Monthly": 1, "Yearly": 2}
        ContractTypeEncoded = np.array([ContractTypeMapping[ct] for ct in ContractType])

        churning = (0.4 + (MonthlySpending / 130) + (Age / 70) - (Tenure / 76)
                    - (SupportCalls / 15) + (ContractTypeEncoded / 12)).astype(int)

        data_dict = {
            "Age": Age,
            "Monthly Spending": MonthlySpending,
            "Tenure": Tenure,
            "ContractType": ContractType,
            "InternetUsage": InternetUsage,
            "SupportCalls": SupportCalls
        }

        self.churn_data = pd.DataFrame(data_dict)
        self.churn_data["Churn"] = churning
        self.churn_data["Churn"] = self.churn_data["Churn"].replace({1:"Yes", 0:"No"})
        self.churn_data = self.churn_data.head(9)

        self.churn_data = self.churn_data.to_html(classes="table text-light", index=False, border=0)

        churn_features = np.c_[Age, MonthlySpending, Tenure, ContractTypeEncoded, InternetUsage, SupportCalls]
        self.x = churn_features
        self.y = churning

        return self.churn_data
    
    def scaling(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        eps = 1e-6
        std = np.where(std == 0, eps, std)
        scaled = (X - mean) / std
        return mean, std, scaled

    def process(self):
        self.mean, self.std, Scaled = self.scaling(self.x)
        self.x_scaled = np.c_[np.ones(Scaled.shape[0]), Scaled]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def costFunction(self, x, y, theta, lam_=0, regularization=None):
        m = len(y)
        hypothesis = self.sigmoid(x.dot(theta))
        # Avoid log(0) by adding epsilon
        epsilon = 1e-8
        cost = -(1 / m) * np.sum(y * np.log(hypothesis + epsilon) +
                                  (1-y) * np.log(1 - hypothesis + epsilon))

        if regularization == 'ridge':
            cost += (lam_ / (2 * m)) * np.sum(theta[1:] ** 2)
        elif regularization == 'lasso':
            cost += (lam_ / (2 * m)) * np.sum(np.abs(theta[1:]))

        return cost

    def gradientdescent(self, x, y, theta, alpha, epoch, lam_=0, regularization=None):
        m = len(y)
        costhistory = []

        for _ in range(epoch):
            hypothesis = self.sigmoid(x.dot(theta))
            diff = hypothesis - y
            gradient = (1 / m) * x.T.dot(diff)

            if regularization == "ridge":
                gradient[1:] += (lam_ / m) * theta[1:]
            elif regularization == "lasso":
                gradient[1:] += (lam_ / m) * np.sign(theta[1:])

            theta -= alpha * gradient
            costhistory.append(self.costFunction(x, y, theta, lam_, regularization))

        return theta, costhistory

    def KFold(self, regularization=None):
        m = len(self.y)
        indices = np.arange(m)
        np.random.shuffle(indices)
        folds = np.array_split(indices, self.k)
        errors = []

        for fold in folds:
            test_index = fold
            train_index = np.setdiff1d(indices, test_index)

            x_train, y_train = self.x_scaled[train_index], self.y[train_index]
            x_test, y_test = self.x_scaled[test_index], self.y[test_index]

            theta = np.zeros(x_train.shape[1])
            theta, _ = self.gradientdescent(x_train, y_train, theta, self.alpha, self.epoch, self.lam_, regularization)

            prediction = self.sigmoid(x_test.dot(theta))
            epsilon = 1e-8
            logloss = -(1 / len(y_test)) * np.sum(
                y_test * np.log(prediction + epsilon) + (1 - y_test) * np.log(1 - prediction + epsilon)
            )
            errors.append(logloss)

        return np.mean(errors)

    def train(self, regularization=None):
        theta = np.zeros(self.x_scaled.shape[1])
        self.theta, self.CostHistory = self.gradientdescent(
            self.x_scaled, self.y, theta, self.alpha, self.epoch, self.lam_, regularization
        )
        return self.theta

    def predict(self, new_data):
        new_data = new_data.reshape(1, -1)
        new_data_scaled = (new_data - self.mean) / self.std
        new_data_normalized = np.c_[np.ones(new_data_scaled.shape[0]), new_data_scaled]
        result = self.sigmoid(new_data_normalized.dot(self.theta))

        if result >= 0.5:
            return "Yes! Customer will churn"
        else:
            return "No! Customer won't churn"
            

    def plot(self):
        if self.CostHistory:
            plt.figure(figsize=(8, 6))
            plt.title("Gradient Descent Convergence")
            plt.plot(range(self.epoch), self.CostHistory)
            plt.ylabel("Cost Function")
            plt.xlabel("Iterations")
            plt.show()



