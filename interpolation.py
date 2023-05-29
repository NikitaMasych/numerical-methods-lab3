import numpy as np
from numpy.polynomial.polynomial import Polynomial


def distribute_points(a, b, n):
    h = (b - a) / n
    result = []
    for i in range(n + 1):
        result.append(a + h * i)
    return np.array(result)


def lagrange(x, y, domain=None):
    if domain is None:
        domain = [-1, 1]

    points_amount = x.shape[0]
    result = Polynomial([0], domain, domain)

    omega_factors = []
    omega_bottom_factors = np.eye(points_amount)

    for i, x_i in enumerate(x):
        omega_factors.append(Polynomial([-x_i, 1], domain, domain))
        for j, x_j in enumerate(x):
            if i != j:
                omega_bottom_factors[i, j] = x_i - x_j

    for i, x_i in enumerate(x):
        bot = np.prod(omega_bottom_factors[i, :])
        temp = y[i] / bot
        for j in range(points_amount):
            if i != j:
                temp *= omega_factors[j]
        result += temp

    return result


def newton(x, y):
    points_amount = x.shape[0]

    div_diffs = np.zeros((points_amount, points_amount))
    div_diffs[:, 0] = y

    for j in range(1, points_amount):
        for i in range(points_amount - j):
            div_diffs[i, j] = (div_diffs[i + 1, j - 1] - div_diffs[i, j - 1]) / (x[i + j] - x[i])

    result = div_diffs[0, 0]

    for i in range(1, points_amount):
        temp = div_diffs[0, i]

        for j in range(i):
            temp *= Polynomial([-x[j], 1])

        result += temp

    return result


class CubicSpline:
    def __init__(self, x, funcs):
        self.ranges = x
        self.funcs = funcs

    def __call__(self, value):
        # Find the appropriate range for the given value
        for i, r in enumerate(self.ranges):
            if r > value:
                return self.funcs[i - 1](value)
        # If value is outside the defined ranges, return None or handle the case accordingly


def tridiagonal_gauss(A: np.ndarray, f: np.ndarray):
    # Gaussian elimination with partial pivoting for tridiagonal system of equations
    c = -np.diag(A)
    a = np.diag(A, k=-1)
    b = np.diag(A, k=1)

    alpha_i = b[0] / c[0]
    f = f.copy().reshape((c.shape[0]))
    beta_i = -f[0] / c[0]

    alpha = [alpha_i]
    beta = [beta_i]
    size = A.shape[0]
    x = np.zeros(size)

    # Forward substitution
    for i in range(1, size - 1):
        z_i = c[i] - alpha_i * a[i - 1]
        beta_i = (-f[i] + a[i - 1] * beta_i) / z_i
        alpha_i = b[i] / z_i
        alpha.append(alpha_i)
        beta.append(beta_i)

    # Back-substitution
    z_i = c[-1] - alpha_i * a[-1]
    x[-1] = (-f[-1] + a[-1] * beta_i) / z_i
    for i in range(size - 1)[::-1]:
        x[i] = x[i + 1] * alpha[i] + beta[i]

    return x


def cubic_spline(x, y, h):
    points_amount = x.shape[0] - 1

    # Construct the tridiagonal matrix A
    A_shape = (points_amount - 1, points_amount - 1)
    A = np.zeros(A_shape)
    A += np.diag([(2 * h) / 3] * (points_amount - 1))
    ul_diags = [h / 6] * (points_amount - 2)
    A += np.diag(ul_diags, -1)
    A += np.diag(ul_diags, 1)

    # Construct the matrix H
    H_shape = (points_amount - 1, points_amount + 1)
    H = np.zeros(H_shape)
    H[:, :points_amount - 1] += np.diag([1 / h] * (points_amount - 1))
    H[:, 1:points_amount] += np.diag([-2 / h] * (points_amount - 1))
    H[:, 2:points_amount + 1] += np.diag([1 / h] * (points_amount - 1))

    # Calculate the right-hand side vector f
    f = H @ y.T

    # Solve the tridiagonal system of equations using Gaussian elimination
    m = tridiagonal_gauss(A, f)

    M = np.zeros(points_amount + 1)
    M[1:-1] = m

    # Initialize the CubicSpline instance
    result = CubicSpline(np.zeros(points_amount), [])
    result.ranges = x.copy()

    # Calculate the coefficients for each cubic polynomial segment
    for i, x_i in enumerate(x):
        if i == 0:
            continue
        a = y[i]
        d = (M[i] - M[i - 1]) / h
        b = (h / 2 * M[i] - h * h / 6 * d + (y[i] - y[i - 1]) / h)
        d_x = Polynomial([-x_i, 1])
        s = a + b * d_x + M[i] / 2 * d_x ** 2 + d / 6 * d_x ** 3
        result.funcs.append(s)

    return result
