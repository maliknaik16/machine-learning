
import numpy as np

class CustomLinearRegression:
    """
    Linear Regression

    @link https://www.statisticshowto.com/probability-and-statistics/regression-analysis/find-a-linear-regression-equation
    """

    def __init__(self):
        # Slope of the line
        self.a = 0

        # The y-intercept
        self.b = 0

    def fit(self, X, y):
        self.X = X.reshape(-1, 1)
        self.y = y.reshape(-1, 1)

        x_sum = self.X.sum()
        y_sum = self.y.sum()

        xy_sum = (self.X * self.y).sum()
        sigxy_sum = (x_sum * y).sum()

        x_sqr = (self.X * self.X).sum()
        y_sqr = (self.y * self.y).sum()

        x_sqr_sum = x_sqr.sum()
        y_sqr_sum = y_sqr.sum()

        n = len(self.X)
        sigmay_sigmax2 = y_sum * x_sqr_sum
        sigmax_sigmaxy = x_sum * xy_sum
        n_sigmax2 = n * x_sqr_sum
        sigmax_sigmax = x_sum * x_sum

        n_sigmaxy = n * xy_sum
        sigmax_sigmay = x_sum * y_sum

        denominator = n_sigmax2 - sigmax_sigmax

        self.a = ( sigmay_sigmax2 - sigmax_sigmaxy ) / denominator
        self.b = ( n_sigmaxy - sigmax_sigmay ) / denominator


    def get_equation(self):
        return 'y = (%.2f + %.2f * x)' % (self.a, self.b)

    def predict(self, X_pred):
        predictions = []

        if type(X_pred) is float or type(X_pred) is int:
            independent = self.a + (self.b * X_pred)

            return independent

        else:
            for x in X_pred:
                independent = self.a + (self.b * x)
                predictions.append(independent)

        return predictions

data = np.array([
    [43, 99],
    [21, 65],
    [25, 79],
    [42, 75],
    [57, 87],
    [59, 81]
])


clr = CustomLinearRegression()

X = data[:, 0].reshape(-1, 1)
y = data[:, 1]

clr.fit(X, y)
print(clr.get_equation())
print(clr.predict([26]))
