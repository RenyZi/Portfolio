import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LoanDefaultRisk:
    def __init__(self, samples = 1000, epoch = 1000, alpha = 0.1, k = 5, lambda_ = 0.01, regularization = None):
        
        # initializing variables
        np.random.seed(42)

        self.samples = samples
        self.epoch = epoch
        self.alpha = alpha
        self.k = k
        self.lambda_ = lambda_
        self.regularization = regularization

        self.theta = None
        self.Costhistory = None
        self.mean = None
        self.std = None
        self.loanData = None
        self.x = None
        self.y = None

        # helper function
        self.generate()
        self.process()

    # dataset
    def generate(self):
        CustomerAge = np.random.randint(20, 71, size=self.samples)
        Annualincome = np.random.randint(50000, 110000, size=self.samples)
        CreditScore = np.random.randint(500, 1001, size=self.samples)
        LoanAmount = np.random.randint(10000, 50001, size=self.samples)
        LoanTerm = np.random.randint(24, 61, size=self.samples)
        IncomeRatio = LoanAmount / LoanTerm

        # Normalize features for linear combination
        age_norm = (CustomerAge - 20) / (70 - 20)
        income_norm = (Annualincome - 50000) / (110000 - 50000)
        credit_norm = (CreditScore - 500) / (1000 - 500)
        loan_ratio_norm = (IncomeRatio - min(IncomeRatio)) / (max(IncomeRatio) - min(IncomeRatio))

        # Create linear risk score: higher score = higher default chance
        risk_score = (
            0.4 * loan_ratio_norm -    # higher monthly payment → more risk
            0.3 * income_norm -        # lower income → more risk
            0.2 * credit_norm -        # lower credit score → more risk
            0.1 * age_norm             # younger borrowers → slightly more risk
        )

        # Threshold based on risk score percentile
        threshold = np.percentile(risk_score, 75)  # top 25% riskiest default
        DefaultRisk = (risk_score > threshold).astype(int)
        
    

        Loan_dataset = {
            "Customer Age":CustomerAge,
            "Annual Income":Annualincome,
            "Credit Score":CreditScore,
            "Loan Amount":LoanAmount,
            "Loan Term":LoanTerm,
            "Income Ratio":IncomeRatio
        }

        self.loanData = pd.DataFrame.from_dict(Loan_dataset)
        self.loanData["Default"] = DefaultRisk
        self.loanData["Default"] = self.loanData["Default"].replace({1: "Risk of Default", 0: "No Default Risk"})

        self.loanData = self.loanData.sample(15).reset_index(drop=True)
        self.loanData = self.loanData.to_html(classes="Table text-light", index=False, border=0)
        # features
        loan_features = np.c_[CustomerAge, Annualincome, CreditScore, LoanAmount, LoanTerm, IncomeRatio]
        self.x = loan_features
        self.y = DefaultRisk

        return self.loanData
    
    # scaling
    def scaling(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        epsilion = 1e-8
        std = np.where(std == 0, epsilion, std)
        scaled = (X - mean) / std
        return mean, std, scaled
    
    # preprocess
    def process(self):
        self.mean, self.std, Scaled = self.scaling(self.x)
        self.x_scaled = np.c_[np.ones(Scaled.shape[0]), Scaled]

    # sigmoid
    def sigmoid(self, z):
        g = 1/(1 + np.exp(-z))
        return g
    
    # loggloss function
    def crossentropy(self, x, y, theta, lambda_=0, regularization=None):
        m = len(y)
        hypothisis = self.sigmoid(x @ theta)
        eps = 1e-6
        cost = -(1/m) * np.sum( y * np.log(hypothisis + eps) + ((1-y) * np.log(1 - hypothisis + eps)) )

        if regularization == 'ridge':
            cost += (lambda_/m) * np.sum(theta[1:]**2)
        elif regularization == 'lasso':
            cost += (lambda_/m) * np.sum(np.abs(theta[1:]))

        return cost
    
    # gradient
    def Gradientdescent(self, x, y, theta, alpha, epoch, lambda_=0, regularization=None):
        m = len(y)
        costhistory = []

        for _ in range(epoch):
            hypothisis = self.sigmoid(x @ theta)
            diff = hypothisis - y
            gradient = (1/m) * x.T.dot(diff)

            if regularization == 'ridge':
                gradient[1:] += (lambda_/m) * theta[1:]
            elif regularization == 'lasso':
                gradient[1:] += (lambda_/m) * np.sign(theta[1:])
            
            theta -= gradient * alpha
            costhistory.append(self.crossentropy(x, y, theta, lambda_, regularization))

        return theta, costhistory
    
    # cross validation
    def K_Folds(self, regularization=None):
        m = len(self.y)
        indices = np.arange(0, m)
        np.random.shuffle(indices)

        folds = np.array_split(indices, self.k)
        Errs = []

        for fold in folds:
            test_idx = fold
            train_idx = np.setdiff1d(indices, test_idx) 

            # train dataet
            x_train, y_train = self.x_scaled[train_idx], self.y[train_idx]
            # test dataset
            x_test, y_test = self.x_scaled[test_idx], self.y[test_idx]

            # train model
            theta = np.zeros(x_train.shape[1])
            theta, _ = self.Gradientdescent(x_train, y_train, theta, self.alpha, self.epoch, self.lambda_, regularization)

            # test model
            predicted = self.sigmoid(x_test @ theta)
            eps = 1e-6
            costy = -(1/m) * np.sum( y_test * np.log(predicted + eps) + ((1-y_test) * np.log(1 - predicted + eps)))
            
            Errs.append(costy)

        return np.mean(Errs)
    
    # train 
    def train(self, regularization=None):
        theta = np.zeros(self.x_scaled.shape[1])
        self.theta, self.Costhistory = self.Gradientdescent(self.x_scaled, self.y, theta, self.alpha, self.epoch, self.lambda_, regularization)
        return self.theta
    
    # prediction
    def prediction(self, new_infor):
        new_infor = new_infor.reshape(1,-1)
        new_infor_scaled = (new_infor - self.mean) / self.std
        new_infor_normalized = np.c_[np.ones(new_infor_scaled.shape[0]), new_infor_scaled]

        predict_result = self.sigmoid(new_infor_normalized @ self.theta)

        if predict_result >= 0.5:
            return "The customer chances of defaulting the loan is highest!"
        else:
            return "The customer will not default the loan"


    # ploting
    def plot(self):
        if self.Costhistory:
            plt.figure(figsize=(8,6))
            plt.plot(range(self.epoch), self.Costhistory)
            plt.title('Cost history converging')
            plt.ylabel("Cross Entropy")
            plt.xlabel("Iterations")
            plt.show()
