import numpy as np


def linear_random_data(sample_size, a, b, x_min, x_max, noise_factor):
    """ Creates a randam data set based on a
    linear function in a given interval

    Args:
        sample_size: number of data points
        a: coefficent of x^0
        b: coefficent of x^1
        x_min: lower bound value range
        x_max: upper bound value range
        noise_factor: strength of noise added to y

    Returns:
        x: array of x values | len(x)==len(y)
        y: array of y values corresponding to x | len(x)==len(y)
    """
    x = np.random.uniform(x_min, x_max, size=sample_size)

    noise = np.random.randn(sample_size) * noise_factor

    y = a + b * x + noise

    return x, y


def linear_hypothesis(theta_0, theta_1):
    """ Combines given arguments in a linear
    equation and returns it as a function

    Args:
        theta_0: first coefficient
        theta_1: second coefficient

    Returns:
        lambda that models a linear function based on theta_0, theta_1 and x
    """
    return lambda x: theta_0 + theta_1 * x


def mse_cost_function(x, y):
    """ Implements MSE cost function as a function
    J(theta_0, theta_1) on given tranings data

    Args:
        x: vector of x values
        y: vector of ground truth values y

    Returns:
        lambda J(theta_0, theta_1) that models the cost function
    """
    m = len(x)

    return lambda theta_0, theta_1: (1 / (2 * m)) * np.sum(
        (theta_0 + theta_1 * x - y) ** 2
    )


def update_theta(x, y, theta_0, theta_1, learning_rate):
    """ Updates learnable parameters theta_0 and theta_1

    The update is done by calculating the partial derivities of
    the cost function including the linear hypothesis. The
    gradients scaled by a scalar are subtracted from the given
    theta values.

    Args:
        x: array of x values
        y: array of y values corresponding to x
        theta_0: current theta_0 value
        theta_1: current theta_1 value
        learning_rate: value to scale the negative gradient

    Returns:
        t0: Updated theta_0
        t1: Updated theta_1
    """
    m = x.shape[0]

    h = linear_hypothesis(theta_0, theta_1)
    prediction = h(x)

    errors = prediction - y

    gradient_theta_0 = np.sum(errors) / m
    gradient_theta_1 = np.sum(errors * x) / m

    new_theta_0 = theta_0 - learning_rate * gradient_theta_0
    new_theta_1 = theta_1 - learning_rate * gradient_theta_1

    return new_theta_0, new_theta_1


def gradient_descent(
    x,
    y,
    iterations=1000,
    learning_rate=0.0001,
    verbose=False,
):
    """ Minimize theta values of a linear model based on MSE cost function

    Args:
        x: vector, x values from the data set
        y: vector, y values from the data set
        iterations: scalar, number of theta updates
        learning_rate: scalar, scales the negative gradient
        verbose: boolean, print addition information

    Returns:
        costs: list oft costs, one value for each iteration.
        t0s: list of theta_0 values, one value for each iteration
        t1s: list of theta_1 values, one value for each iteration
    """
    theta_0 = 0.0
    theta_1 = 0.0

    costs = []
    t0s = []
    t1s = []

    J = mse_cost_function(x, y)

    for i in range(iterations):
        theta_0, theta_1 = update_theta(x, y, theta_0, theta_1, learning_rate)

        cost = J(theta_0, theta_1)

        costs.append(cost)
        t0s.append(theta_0)
        t1s.append(theta_1)

        if verbose and i % (iterations // 10) == 0:
            print(
                f"Iteration {i}: Cost = {cost:.4f}, theta_0 = {
                    theta_0:.4f}, theta_1 = {theta_1:.4f}"
            )

    return costs, t0s, t1s


def optimize_learning_rate(learning_rates, x, y, iterations):
    """ Train the model for multiple learning
    rates and return all cost histories.

    Args:
        x: Input feature array
        y: Target values
        learning_rates: List or array of learning rates to test
        iterations: Number of gradient descent iterations

    Returns:
        List of cost history lists, one for each learning rate
    """
    cost_histories = []
    for lr in learning_rates:
        costs, _, _ = gradient_descent(x, y, iterations, lr)
        cost_histories.append(costs)

    return cost_histories
