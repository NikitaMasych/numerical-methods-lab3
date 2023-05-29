import numpy as np
import matplotlib.pyplot as plt
import interpolation


def function(x):
    return np.cos(x + 3)


if __name__ == "__main__":
    # Define the interval and number of points for interpolation
    a = -9
    b = 9
    n = 13

    # Generate the x-coordinates for interpolation
    x = interpolation.distribute_points(a, b, n)

    # Perform Lagrange polynomial interpolation
    L = interpolation.lagrange(x, function(x))

    # Perform Newton polynomial interpolation
    N = interpolation.newton(x, function(x))

    # Calculate the step size for cubic spline interpolation
    h = (b - a) / n

    # Perform cubic spline interpolation
    S = interpolation.cubic_spline(x, function(x), h)

    # Set up data points for plotting
    step = 0.1
    graph_padding = 1
    D = np.arange(a - graph_padding, b + graph_padding + step, step)

    # Generate cubic spline interpolation points
    spline_D = np.arange(a, b + step, step)
    spline_result = [S(d) for d in spline_D]

    # Create the plot
    plt.scatter(x, function(x))  # Plot the original data points

    original, = plt.plot(D, function(D))  # Plot the original function
    lagrange, = plt.plot(D, L(D))  # Plot the Lagrange interpolation
    newton, = plt.plot(D, N(D))  # Plot the Newton interpolation
    spline, = plt.plot(spline_D, spline_result)  # Plot the cubic spline interpolation

    # Set labels for the plot
    original.set_label("Original")
    lagrange.set_label("Lagrange")
    newton.set_label("Newton")
    spline.set_label("Spline")

    plt.grid(visible=True)  # Add grid lines to the plot
    plt.legend()  # Display the legend
    plt.show()  # Show the plot
