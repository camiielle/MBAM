import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mbam.geodesic import Geodesic
from mbam.finite_difference import Avv_func, AvvCD4, jacobian_func
from mbam.utils import initial_velocity
from collections import namedtuple

# Six parameters initial values
# Using typical values calibrated for the CMIP5 models mentioned in the article​
# https://journals.ametsoc.org/view/journals/clim/26/6/jcli-d-12-00196.1.xml

# C = 8 W⋅yr/m²/K     # Upper layer heat capacity
# C0 = 100 W⋅yr/m²/K  # Lower layer heat capacity
# lambda_ = 1.3/8 y-1
# gamma = 0.7/8 yr^-1
# gamma0 = 0.7/100 yr^-1
# epsilon = 1        # Efficacy factor
# F = 3.9/8 K/yr     # External forcing for a 4x CO2 increase divided by C

# I make some assumptions to simplify the model:
# 1) efficacy = 1 fixed (so one less parameter)
# 2) Forcing is a STEP FORCING
# 3) Tx(0)=0 as initial conditions for x=a,o
# under these assumptions an analytical solution exists and takes a simple form

M = 6  # number of predictions
N = 4  # number of parameters

t = np.array([1., 5., 10.])  # yr

# heat capacities and gamma factors only allow for positive values, so i defined them as square of real parameters

def r(x):
    for i in range(1, 3):  # ensuring gammas stay positive
        if x[i] < 0:
            x[i] = 1e-10
    λ, γ, γ0, F = x[0], x[1], x[2], x[3]

    # General parameters
    b = λ + γ + γ0
    b_star = λ + γ - γ0
    delta = b*b - 4 * λ * γ0

    # Mode parameters (Fast and Slow)
    τ_f = (b - np.sqrt(delta)) / (2 * γ0 * λ)
    τ_s = (b + np.sqrt(delta)) / (2 * γ0 * λ)

    φ_s = 1 / (2 * γ) * (b_star + np.sqrt(delta))
    φ_f = 1 / (2 * γ) * (b_star - np.sqrt(delta))

    a_f = φ_s * τ_f * λ / (φ_s - φ_f)
    l_f = a_f * τ_f * λ / (1 + γ/γ0)

    a_s = -φ_f * τ_s * λ / (φ_s - φ_f)
    l_s = a_s * τ_s * λ / (1 + γ/γ0)

    # defining ODEs solutions
    Ta = F/λ*(a_f*(1-np.exp(-1/τ_f*t)) + a_s*(1-np.exp(-1/τ_s*t)))
    To = F/λ*(a_f*φ_f*(1-np.exp(-1/τ_f*t)) + φ_s*a_s*(1-np.exp(-1/τ_s*t)))

    return np.hstack((Ta, To))

# Jacobian (computed numerically)
def j(x):
    jacob = jacobian_func(r, M, N)
    return jacob(x)

# Directional second derivative
def Avv(x, v):
    avv_function = Avv_func(r)
    return avv_function(x, v)

# Choose starting parameters
x = [1.3/8, 0.2125, 0.007, 0.4875]
v = initial_velocity(x, j, Avv)

# Callback
def callback(g):
    # Integrate until the norm of the velocity has grown by a factor of 10
    # and print out some diagnotistic along the way
    _, s, _ = np.linalg.svd(j(g.xs[-1]))
    print(
        "Iteration: %i, tau: %f, |v| = %f, eigenvalue: %.8f"
        % (len(g.vs), g.ts[-1], np.linalg.norm(g.vs[-1]), s[-1])
    )
    current_parameters = g.xs[-1]
    for i in range(1, 3):
        if current_parameters[i] < 0:
            current_parameters[i] = 1e-10
    return np.linalg.norm(g.vs[-1]) < 1200


# Construct the geodesic
geo = Geodesic(r, j, Avv, x, v, atol=1e-2, rtol=1e-2,
               parameterspacenorm=False, callback=callback)

# Integrate
geo.integrate(480)

# Plot the geodesic path to find the limit
colors = ['r', 'g', 'b', 'orange']
labels = ['λ', 'γ', 'γ0', 'F']
parameter_values = [geo.xs[:, 0], geo.xs[:, 1], geo.xs[:, 2], geo.xs[:, N-1]]

for i in range(N):
    plt.plot(geo.ts, parameter_values[i], label=labels[i], color=colors[i])
plt.xlabel("Tau")
plt.ylabel("Parameter Values")
plt.legend()
plt.show()

# The first boundary we find is for F-> infinity
# define new model (M=6,N=2)
N_new = 2


def r_new(y):
    λ = 1e10
    γ = 1e10
    γ0, F = y[0], y[1]

    # General parameters
    b = λ + γ + γ0
    b_star = λ + γ - γ0
    delta = b*b - 4 * λ * γ0

    # Mode parameters (Fast and Slow)
    τ_f = (b - np.sqrt(delta)) / (2 * γ0 * λ)
    τ_s = (b + np.sqrt(delta)) / (2 * γ0 * λ)

    φ_s = 1 / (2 * γ) * (b_star + np.sqrt(delta))
    φ_f = 1 / (2 * γ) * (b_star - np.sqrt(delta))

    a_f = φ_s * τ_f * λ / (φ_s - φ_f)
    l_f = a_f * τ_f * λ / (1 + γ/γ0)

    a_s = -φ_f * τ_s * λ / (φ_s - φ_f)
    l_s = a_s * τ_s * λ / (1 + γ/γ0)

    # defining ODEs solutions
    Ta = F/λ*(a_f*(1-np.exp(-1/τ_f*t)) + a_s*(1-np.exp(-1/τ_s*t)))
    To = F/λ*(a_f*φ_f*(1-np.exp(-1/τ_f*t)) + φ_s*a_s*(1-np.exp(-1/τ_s*t)))

    return np.hstack((Ta, To))


def j_new(y):
    jacob = jacobian_func(r_new, M, N_new)
    return jacob(y)

# Directional second derivative


def Avv_new(y, v_new):
    avv_function = Avv_func(r_new)
    return avv_function(y, v_new)


# Choose starting parameters
y = [0.7/100, 3.9/8]
print(r_new(y))
print(j_new(y))
v_new = initial_velocity(y, j_new, Avv_new)

# Callback


def callback_new(g):
    # Integrate until the norm of the velocity has grown by a factor of 10
    # and print out some diagnotistic along the way
    _, s, _ = np.linalg.svd(j_new(g.xs[-1]))
    print(
        "Iteration: %i, tau: %f, |v| = %f, eigenvalue: %.25f"
        % (len(g.vs), g.ts[-1], np.linalg.norm(g.vs[-1]), s[-1])
    )
    return True  # np.linalg.norm(g.vs[-1]) < 80.0


# Construct the geodesic
geo_new = Geodesic(r_new, j_new, Avv_new, y, v_new, atol=1e-2, rtol=1e-2,
                   parameterspacenorm=False, callback=callback_new)

# Integrate
geo_new.integrate(480)

# plot_geodesic_path(geo_new, colors[2:], labels[2:], N_new)

# define new model (M=6,N=1)
N_new_new = 1


def r_new_new(F):
    λ = 1e10
    γ = 1e10
    γ0 = 1e10

    # General parameters
    b = λ + γ + γ0
    b_star = λ + γ - γ0
    delta = b*b - 4 * λ * γ0

    # Mode parameters (Fast and Slow)
    τ_f = (b - np.sqrt(delta)) / (2 * γ0 * λ)
    τ_s = (b + np.sqrt(delta)) / (2 * γ0 * λ)

    φ_s = 1 / (2 * γ) * (b_star + np.sqrt(delta))
    φ_f = 1 / (2 * γ) * (b_star - np.sqrt(delta))

    a_f = φ_s * τ_f * λ / (φ_s - φ_f)
    l_f = a_f * τ_f * λ / (1 + γ/γ0)

    a_s = -φ_f * τ_s * λ / (φ_s - φ_f)
    l_s = a_s * τ_s * λ / (1 + γ/γ0)

    # defining ODEs solutions
    Ta = F/λ*(a_f*(1-np.exp(-1/τ_f*t)) + a_s*(1-np.exp(-1/τ_s*t)))
    To = F/λ*(a_f*φ_f*(1-np.exp(-1/τ_f*t)) + φ_s*a_s*(1-np.exp(-1/τ_s*t)))

    return np.hstack((Ta, To))


def j_new_new(F):
    jacob = jacobian_func(r_new_new, M, N_new_new)
    return jacob(F)

# Directional second derivative


def Avv_new_new(F, v_new_new):
    avv_function = Avv_func(r_new_new)
    return avv_function(F, v_new_new)


# Choose starting parameters
F = np.array([3.9/8])
v_new_new = initial_velocity(F, j_new_new, Avv_new_new)

# Callback


def callback_new_new(g):
    # Integrate until the norm of the velocity has grown by a factor of 10
    # and print out some diagnotistic along the way
    _, s, _ = np.linalg.svd(j_new_new(g.xs[-1]))
    print(
        "Iteration: %i, tau: %f, |v| = %f, eigenvalue: %.25f"
        % (len(g.vs), g.ts[-1], np.linalg.norm(g.vs[-1]), s[-1])
    )
    return True


# Construct the geodesic
geo_new_new = Geodesic(r_new_new, j_new_new, Avv_new_new, F, v_new_new, atol=1e-2, rtol=1e-2,
                       parameterspacenorm=False, callback=callback_new_new)

# Integrate
geo_new_new.integrate(480)

# plot_geodesic_path(geo_new, colors[3:], labels[3:], N_new_new)
