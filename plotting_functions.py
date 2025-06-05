import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from linear_regression import linear_hypothesis, mse_cost_function


def plot_data_with_hypothesis(x, y, theta_0, theta_1):
    """ Plots the data (x, y) together with a
    hypothesis given theta0 and theta1.
    """
    h = linear_hypothesis(theta_0, theta_1)

    x_line = np.linspace(np.min(x), np.max(x), 100)
    y_line = h(x_line.astype(np.float64))

    _ = plt.scatter(x, y, color="red", label="Data points")

    _ = plt.plot(
        x_line,
        y_line,
        color="blue",
        label=f"Hypothesis: h(x) = {theta_0:.2f} + {theta_1:.2f}*x",
    )

    cost_fn = mse_cost_function(x, y)
    cost = cost_fn(theta_0, theta_1)
    plt.legend()
    plt.show()
    print(f"MSE Cost for θ0={theta_0}, θ1={theta_1}: {cost:.4f}")


def plot_cost_grid(
    cost,
    interval,
    num_samples,
    theta0_offset=0.,
    theta1_offset=0.,
):
    """ Creates mesh points for a 3D plot
    based on a given interval and a cost function.
    The function creates a numpy meshgrid for plotting
    a 3D-plot of the cost function. Additionally, for
    the mesh grid points cost values are calulated and returned.

    Args:
        cost: a function that is used to calculate costs.
              The function "cost" was typically e.g.
              created by calling "cost = mse_cost_function(x, y)".
              So, the data x,y and the model are used internally in cost.
              The arguments of the function cost are theta_0 und theta_1,
              i.e. cost(theta_0, theta_0).
        interval: a scalar that defines the range
                  [-interval, interval] of the mesh grid
        num_mesh: the total number of points in the
                  mesh grid is num_mesh * num_mesh (equaly distributed)
        theta0_offset: shifts the plotted interval for theta0 by a scalar
        theta1_offset: shifts the plotted interval for theta1 by a scalar

    Returns:
        t0: a matrix representing a meshgrid for the
            values of the first plot dimesion (Theta 0)
        t1: a matrix representing a meshgrid for the
            values of the second plot dimesion (Theta 1)
        c: a matrix respresenting cost values (third plot dimension)
    """
    t0_vals = np.linspace(
        -interval + theta0_offset, interval + theta0_offset, num_samples
    )
    t1_vals = np.linspace(
        -interval + theta1_offset, interval + theta1_offset, num_samples
    )

    t0, t1 = np.meshgrid(t0_vals, t1_vals)

    vectorized_cost = np.vectorize(cost)

    return t0, t1, vectorized_cost(t0, t1)


def plot_cost(t0, t1, c):
    """ Creates a counter and a surface plot based on given data

    Args:
        T0: a matrix representing a meshgrid for X values (Theta 0)
        T1: a matrix representing a meshgrid for Y values (Theta 1)
        C: a matrix respresenting cost values
    """
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    contour = ax1.contourf(t0, t1, c, levels=50, cmap="viridis")
    fig.colorbar(contour, ax=ax1)
    ax1.set_title("Contour Plot of Cost Function")
    ax1.set_xlabel(r"$\theta_0$")
    ax1.set_ylabel(r"$\theta_1$")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    assert isinstance(ax2, Axes3D)

    ax2.plot_surface(t0, t1, c, cmap="viridis", edgecolor="none")
    ax2.set_title("Surface Plot of Cost Function")
    ax2.set_xlabel(r"$\theta_0$")
    ax2.set_ylabel(r"$\theta_1$")
    ax2.set_zlabel("Cost")

    plt.tight_layout()
    plt.show()


def plot_evaluation(cost_hist, theta_0, theta_1, x, y):
    """ Plots a cost curve and the decision boundary

    The Method plots a cost curve from a given training process (cost_hist).
    It also plots the data set (x,y) and draws a linear decision boundary
    with the parameters theta_0 and theta_1 into the plotted data set.

    Args:
        cost_hist: vector, history of all cost values from a opitmization
        theta_0: scalar, model parameter for boundary
        theta_1: scalar, model parameter for boundary
        x: vector, x values from the data set
        y: vector, y values from the data set
    """
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(cost_hist, label="Cost")
    plt.title("Cost over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Cost (MSE)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(x, y, color="blue", label="Data")

    x_vals = np.linspace(min(x), max(x), 100)
    y_vals = theta_0 + theta_1 * x_vals
    plt.plot(x_vals, y_vals, color="red", label="Decision Boundary")

    plt.title("Data and Linear Fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_learning_rate_comparison(
    learning_rates,
    cost_histories,
    log_scale: bool = False,
):
    """ Plot cost histories for different learning rates

    Args:
        learning_rates: list of learning rates used
        cost_histories: list of lists, each inner list
        contains cost history for one learning rate
    """
    plt.figure(figsize=(10, 6))

    for lr, history in zip(learning_rates, cost_histories):
        plt.plot(history, label=f"lr={lr}")

    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    if log_scale:
        plt.yscale("log")
    plt.title("Learning Rate Comparison (log-scaled cost)")
    plt.legend()
    plt.tight_layout()
    plt.show()
