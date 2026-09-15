import numpy as np


# =========================================================
# HYPOTHESIS
# =========================================================

def hypothesis(X, theta):
    """
    h_theta(X) = X theta
    """

    return X @ theta


# =========================================================
# COST FUNCTION
# =========================================================

def cost(X, y, theta):
    """
    J(theta) = 1/2 * sum((prediction - y)^2)
    """

    predictions = hypothesis(X, theta)

    errors = predictions - y

    return 0.5 * np.sum(errors ** 2)


# =========================================================
# NORMAL EQUATION
# =========================================================

def fit_normal(X, y):
    """
    theta = (X^T X)^-1 X^T y
    """

    XtX = X.T @ X

    XtX_inverse = np.linalg.inv(XtX)

    theta = (
        XtX_inverse
        @ X.T
        @ y
    )

    return theta


# =========================================================
# BATCH GRADIENT DESCENT
# =========================================================

def fit_batch_gd(
    X,
    y,
    alpha,
    n_iters
):

    # Start theta at zero
    theta = np.zeros(
        X.shape[1],
        dtype=float
    )


    history = []


    for iteration in range(n_iters):

        # Current predictions
        predictions = hypothesis(
            X,
            theta
        )


        # Error
        errors = y - predictions


        # Gradient/update
        update = (
            alpha *
            (X.T @ errors)
        )


        theta = theta + update


        # Record cost AFTER update
        current_cost = cost(
            X,
            y,
            theta
        )


        history.append(
            current_cost
        )


    return theta, np.array(history)


# =========================================================
# STOCHASTIC GRADIENT DESCENT
# =========================================================

def fit_sgd(
    X,
    y,
    alpha,
    n_epochs
):

    theta = np.zeros(
        X.shape[1],
        dtype=float
    )


    history = []


    n = X.shape[0]


    for epoch in range(n_epochs):

        # Loop through training rows
        for i in range(n):

            xi = X[i]

            yi = y[i]


            # Prediction for one row
            prediction = (
                xi @ theta
            )


            # Error
            error = (
                yi - prediction
            )


            # SGD update
            theta = (
                theta
                + alpha * error * xi
            )


        # Cost after complete epoch
        current_cost = cost(
            X,
            y,
            theta
        )


        history.append(
            current_cost
        )


    return theta, np.array(history)


# =========================================================
# RMSE
# =========================================================

def rmse(y_true, y_pred):

    return np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


# =========================================================
# CLIP NEGATIVE PREDICTIONS
# =========================================================

def clip_predictions(predictions):

    return np.maximum(
        predictions,
        0
    )