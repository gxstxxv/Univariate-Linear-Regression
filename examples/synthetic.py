import numpy as np
from plotting_functions import (
    plot_data_with_hypothesis,
    plot_cost_grid,
    plot_cost,
    plot_evaluation,
    plot_learning_rate_comparison,
)
from linear_regression import (
    linear_random_data,
    mse_cost_function,
    gradient_descent,
    optimize_learning_rate,
)


def main():
    np.random.seed(seed=42)

    x, y = linear_random_data(
        sample_size=200, a=2.0, b=-5.0, x_min=-10, x_max=10, noise_factor=10
    )

    t1 = 1
    t0 = 0
    plot_data_with_hypothesis(x, y, theta_0=t0, theta_1=t1)

    j = mse_cost_function(x, y)
    t0, t1, c = plot_cost_grid(
        j, interval=1000, num_samples=51, theta1_offset=5.0
    )
    plot_cost(t0, t1, c)

    cost_hist, t0_hist, t1_hist = gradient_descent(
        x, y, iterations=250, learning_rate=0.0003, verbose=True
    )

    plot_evaluation(cost_hist, t0_hist[-1], t1_hist[-1], x, y)

    potential_lr = [0.0001, 0.0007, 0.001, 0.007,
                    0.01, 0.015, 0.02, 0.028, 0.03, 0.04]
    cost_histories = optimize_learning_rate(potential_lr, x, y, 50)
    plot_learning_rate_comparison(potential_lr, cost_histories)


if __name__ == "__main__":
    main()
