# Univariate Linear Regression

A simple implementation of univariate linear regression using gradient descent, built with NumPy and Matplotlib.

## Contents

1. **Data Generation**: Creates synthetic data with linear relationship y = a + bx + noise
2. **Cost Function**: Calculates Mean Squared Error between predictions and true values
3. **Gradient Descent**: Iteratively finds optimal parameters θ₀ and θ₁
4. **Visualization**: Shows the entire learning process through various plots

## Generated Plots

### 1. Initial Hypothesis vs. Data

![figure_1](https://github.com/gxstxxv/Univariate-Lineare-Regression/blob/main/plots/Figure_1.png)
Shows the raw data points and an initial line estimate.

### 2. 3D Cost Function

![figure_2](https://github.com/gxstxxv/Univariate-Lineare-Regression/blob/main/plots/Figure_2.png)
Visualizes the cost function as 3D surface and contour plot - shows the "valley" where the minimum lies.

### 3. Learning Progress and Final Fit

![figure_3](https://github.com/gxstxxv/Univariate-Lineare-Regression/blob/main/plots/Figure_3.png)
Left: Cost reduction over iterations. Right: Final line fit to the data.

### 4. Learning Rate Comparison

![figure_4](https://github.com/gxstxxv/Univariate-Lineare-Regression/blob/main/plots/Figure_4.png)
Compares different learning rates and shows their impact on convergence speed.
