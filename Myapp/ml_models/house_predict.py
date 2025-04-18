import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class HousePricePredictor:
    def __init__(self, house_samples=500, alpha=0.1, epoch=1000, k=5, lambda_=1):
        np.random.seed(42)
        self.house_samples = house_samples
        self.alpha = alpha
        self.epoch = epoch
        self.k = k
        self.lambda_ = lambda_

        self.mean = None
        self.std = None
        self.theta = None
        self.cost_history = None
        self.house_details = None
        self.y = None
        self.X = None

        self.generate_data()
        self.preprocess()

    def generate_data(self):
        # Feature creation
        sf = np.random.randint(500, 5000, size=self.house_samples)
        bd = np.clip(sf // 1000 + np.random.randint(0, 2, size=self.house_samples), 1, 5)
        ba = np.clip(bd + np.random.randint(0, 2, size=self.house_samples), 1, 4)
        age = np.random.randint(1, 20, size=self.house_samples)
        crime = np.random.uniform(1.0, 9.9, size=self.house_samples)
        dist = np.random.randint(1, 25, size=self.house_samples)

        self.price = (
            490.91
            + 0.10387 * sf
            + 0.96411 * bd
            + 1.00699 * ba
            - 0.22552 * age
            + 2.08973 * crime
            - 0.58644 * dist
        )

        self.house_details = pd.DataFrame({
            "Square_feet": sf,
            "Bedrooms": bd,
            "Bathrooms": ba,
            "Age": age,
            "Crime_rate": crime,
            "Distance_city": dist
        })
        self.house_details["Price($ 1000)"] = self.price
        self.house_details = self.house_details.sample(9).reset_index(drop=True)

        sample_housedataset = self.house_details.to_html(classes="table text-light", index=False, border=0)

        features = np.c_[sf, bd, ba, age, crime, dist]
        self.X = features
        self.y = self.price

        return sample_housedataset
        

    def scale_features(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std == 0, 1e-6, std)
        scaled = (X - mean) / std
        return mean, std, scaled

    def preprocess(self):
        self.mean, self.std, scaled = self.scale_features(self.X)
        self.X_scaled = np.c_[np.ones(scaled.shape[0]), scaled]  # Add bias term

    def cost_function(self, X, y, theta, lambda_=0, regularization=None):
        m = len(y)
        h = X.dot(theta)
        error = h - y
        cost = (1 / (2 * m)) * np.sum(error**2)

        if regularization == 'ridge':
            cost += (lambda_ / (2 * m)) * np.sum(theta[1:]**2)
        elif regularization == 'lasso':
            cost += (lambda_ / (2 * m)) * np.sum(np.abs(theta[1:]))

        return cost

    def gradient_descent(self, X, y, theta, alpha, epoch, lambda_=0, regularization=None):
        m = len(y)
        cost_history = []

        for _ in range(epoch):
            h = X.dot(theta)
            error = h - y
            gradient = (1 / m) * X.T.dot(error)

            if regularization == 'ridge':
                gradient[1:] += (lambda_ / m) * theta[1:]
            elif regularization == 'lasso':
                gradient[1:] += (lambda_ / m) * np.sign(theta[1:])

            theta -= alpha * gradient
            cost = self.cost_function(X, y, theta, lambda_, regularization)
            cost_history.append(cost)

        return theta, cost_history

    def k_fold_cross_validation(self, regularization=None):
        m = len(self.y)
        indices = np.arange(m)
        np.random.shuffle(indices)
        folds = np.array_split(indices, self.k)
        errors = []

        for fold in folds:
            test_idx = fold
            train_idx = np.setdiff1d(indices, test_idx)

            x_train, y_train = self.X_scaled[train_idx], self.y[train_idx]
            x_test, y_test = self.X_scaled[test_idx], self.y[test_idx]

            theta = np.zeros(x_train.shape[1])
            theta, _ = self.gradient_descent(x_train, y_train, theta, self.alpha, self.epoch, self.lambda_, regularization)

            preds = x_test.dot(theta)
            error = np.mean((preds - y_test) ** 2)
            errors.append(error)

        return np.mean(errors)

    def train(self, regularization=None):
        theta = np.zeros(self.X_scaled.shape[1])
        self.theta, self.cost_history = self.gradient_descent(
            self.X_scaled, self.y, theta, self.alpha, self.epoch, self.lambda_, regularization
        )
        return self.theta

    def predict(self, new_data):
        new_data = new_data.reshape(1,-1)
        scaled = (new_data - self.mean) / self.std
        normalized = np.c_[np.ones(scaled.shape[0]), scaled]
        return normalized.dot(self.theta)

    def plot_cost(self):
        if self.cost_history:
            plt.plot(range(self.epoch), self.cost_history)
            plt.title("Cost Function Over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Cost")
            plt.grid(True)
            plt.show()


