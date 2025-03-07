class GradientDescent:
    # > calculate the gradient of the cost function mse
    # > formula: dJ(w,b)/dw, dJ(w,b)/db
    @staticmethod
    def gradient(w, b, x, y):
        d_w = 0
        d_b = 0
        m = x.shape[0]

        for i in range(m):
            prediction = w * x[i] + b
            d_w += (prediction - y[i]) * x[i]
            d_b += (prediction - y[i])

        d_w = d_w / m
        d_b = d_b / m
        return d_w, d_b
    
    # > gradient descent
    # > formula: w - a * d_w, b - a * d_b
    @staticmethod
    def call(w_init, b_init, alpha, iters, x, y, mse, predictor):
        w = w_init
        b = b_init
        history = []

        #run until iterations are done
        for i in range(iters):
            d_w,d_b = GradientDescent.gradient(w, b, x, y)
            w = w - alpha * d_w
            b = b - alpha * d_b

            #store the cost per iteration
            predictions = predictor.predict(w, b, x)
            cost = mse(predictions, x, y)
            history.append(cost)

        return w, b, history

