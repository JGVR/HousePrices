import numpy as np
import copy

class GradientDescent:
    # > calculate the gradient of the cost function mse
    # > formula: dJ(w,b)/dw, dJ(w,b)/db
    @staticmethod
    def gradient(w, b, X, y):
        m,n = X.shape
        dj_w = np.zeros((n,))
        dj_b = 0

        for i in range(m):
            err = (np.dot(X[i], w) + b) - y[i]
            for j in range(n):
                dj_w[j] += err * X[i,j]
            dj_b += err

        dj_w = dj_w / m
        dj_b = dj_b / m
        return dj_w, dj_b
    
    # > gradient descent
    # > formula: w - a * d_w, b - a * d_b
    @staticmethod
    def call(w_init, b_init, alpha, iters, X, y, mse):
        w = copy.deepcopy(w_init)
        b = b_init
        history = []

        #run until iterations are done
        for i in range(iters):
            dj_w,dj_b = GradientDescent.gradient(w, b, X, y)
            w = w - alpha * dj_w
            b = b - alpha * dj_b

            #store the cost per iteration
            cost = mse.call(w, b, X, y)
            history.append(cost)

        return w, b, history

