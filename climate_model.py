import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mbam.geodesic import Geodesic
from mbam.finite_difference import Avv_func, AvvCD4
from mbam.utils import initial_velocity
from collections import namedtuple

# Six parameters initial values
# Using typical values calibrated for the CMIP5 models mentioned in the article​
# https://journals.ametsoc.org/view/journals/clim/26/6/jcli-d-12-00196.1.xml

# C = 8           # Upper layer heat capacity
# C0 = 100        # Lower layer heat capacity
# lambda_ = 1.3/8
# gamma = 0.7/8
# gamma0 = 0.7/100
# epsilon = 1      # Efficacy factor
# F = 3.9/8        # External forcing for a 4x CO2 increase divided by C

# I make some assumptions to simplify the model:
# 1) efficacy = 1 fixed (so one less parameter)
# 2) Forcing is a STEP FORCING
# 3) Tx(0)=0 as initial conditions for x=a,o
# under these assumptions an analytical solution exists and takes a simple form


Parameters = namedtuple('Parameters', ['λ', 'γ', 'γ0', 'F'])

t = np.array([0.5, 1.0, 2.0])

def r(x):
    λ, γ, γ0, F = x.λ, x.γ, x.γ0, x.F

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

x = Parameters(λ=1.3/8, γ=0.7/8, γ0=0.7/100, F=3.9/8)

# Jacobian
def j(x):
    return

# Directional second derivative


def Avv(x, v):
    return

# Choose starting parameters
x = ...
# v = initial_velocity(x, j, Avv)

# standard deviations
sigma_a = 0.1
sigma_o = 0.1
