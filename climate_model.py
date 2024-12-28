import matplotlib.pyplot as plt
import numpy as np
from mbam.geodesic import Geodesic
from mbam.finite_difference import Avv_func, AvvCD4, jacobian_func
from mbam.utils import initial_velocity
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh

"""
Multimodel means of values calibrated for the CMIP5 models mentioned in the article​
https://journals.ametsoc.org/view/journals/clim/26/6/jcli-d-12-00196.1.xml

C = 7.3 W⋅yr/m²/K     # Upper layer heat capacity
C0 = 91 W⋅yr/m²/K     # Lower layer heat capacity
lambda_ = 1.13/7.3 y-1
gamma = 0.74/7.3 yr^-1
gamma0 = 0.74/91 yr^-1
F = 6.9/7.3 K/yr     # External forcing for a 4x CO2 increase divided by C

We make some assumptions to simplify the model:
1) efficacy = 1 fixed (so one less parameter)
2) Forcing is a STEP FORCING
3) Tx(0)=0 as initial conditions for x=a,o
under these assumptions an analytical solution exists and takes a simple form

We enforce x_i > 0 by going to log parameters

We adopt the convention that the model has N parameters and makes M
predictions. Then, the output of r(x) should be a vector length M the
output of j(x) (i.e., jacobian) should be an M times N matrix.
The output of Avv(x,v) should be a vector of length M.
"""
np.random.seed(42)

M = 10  # number of predictions
N = 4  # number of parameters

t = np.array([0.01, 0.1, 1., 10., 100.])  # yr
# t = np.array([1.,5.,10.])


def r(x):
    λ, γ, γo, F = np.exp(x)
    if λ == 0:
        λ = 1e-8
    # General parameters
    b = λ + γ + γo
    b_star = λ + γ - γo
    delta = b*b - 4 * λ * γo

    # Mode parameters (Fast and Slow)
    τ_f = (b - np.sqrt(delta)) / (2 * γo * λ)
    τ_s = (b + np.sqrt(delta)) / (2 * γo * λ)

    φ_s = 1 / (2 * γ) * (b_star + np.sqrt(delta))
    φ_f = 1 / (2 * γ) * (b_star - np.sqrt(delta))

    a_f = φ_s * τ_f * λ / (φ_s - φ_f)
    l_f = a_f * τ_f * λ / (1 + γ/γo)

    a_s = -φ_f * τ_s * λ / (φ_s - φ_f)
    l_s = a_s * τ_s * λ / (1 + γ/γo)

    if τ_f == 0:
        τ_f = 1e-8
    if τ_s == 0:
        τ_s = 1e-8

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

def return_sloppiest_eigendirection(g):
    eigenvalues, eigenvectors = np.linalg.eigh(g)
    
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    sloppiest_eigendirection = eigenvectors[:, 0]

    return sloppiest_eigendirection


    

# Choose starting parameters
X = [1.13/7.3, 0.74/7.3, 0.74/91, 6.9/7.3]  # λ, γ, γo, F
x = np.log(X)
print("Sum of predictions squared is:")
print(np.sum(r(x)**2))
v = initial_velocity(x, j, Avv)
noisy_data_point = r(x) + np.random.normal(0., 0.02*r(x), size=r(x).shape)
#noisy_data_point = r(x)
print(f"Noisy point:{noisy_data_point}")

# Callback

def callback(g):
    # Integrate until the norm of the velocity has grown by a factor of 10
    # and print out some diagnostic along the way
    _, s, _ = np.linalg.svd(j(g.xs[-1]))
    print(
        "Iteration: %i, tau: %f, |v| = %f, eigenvalue: %.20f"
        % (len(g.vs), g.ts[-1], np.linalg.norm(g.vs[-1]), s[-1])
    )
    return np.linalg.norm(g.vs[-1]) < 240
    # return np.linalg.norm(g.vs[-1]) < 1200


# Construct the geodesic
geo = Geodesic(r, j, Avv, x, v, atol=1e-2, rtol=1e-2,
               parameterspacenorm=False, callback=callback)

# Integrate
geo.integrate(480, maxsteps=3000)

# Plot the geodesic path to find the limit
colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"] #colorblind friendly
labels = ['logλ', 'logγ', 'logγo', 'logF']


def plot_geodesic_path(geo, colors, labels, N):
    for i in range(N):
        plt.plot(geo.ts, geo.xs[:, i], label=labels[i], color=colors[i])
    plt.xlabel("Tau", fontsize=16)
    plt.ylabel("Parameter Values", fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.show()


plot_geodesic_path(geo, colors, labels, N)


j_i= j(x)
g_i = j_i.T @ j_i 

sloppiest_eigendirection = return_sloppiest_eigendirection(g_i)
indices = np.arange(len(sloppiest_eigendirection))
plt.figure(figsize=(10, 5))
plt.bar(indices, sloppiest_eigendirection, color='blue')
plt.axhline(0, color='gray', linewidth=0.5)  
plt.title("Sloppiest Eigendirection at initial point", fontsize=25, color='black', backgroundcolor='white', pad=20)
plt.xticks(indices, labels, rotation=45, ha='right')  #
plt.xlabel("Bare Parameter", fontsize=18)
plt.ylabel("Bare Parameter Component", fontsize=18)
plt.tick_params(axis='both', labelsize=16)
plt.gca().set_facecolor("white") 
plt.gcf().patch.set_facecolor("white")  

# Display the plot
plt.tight_layout() 
plt.show()


j_final= j(geo.xs[-1])
g_final = j_final.T @ j_final 
final_sloppiest_eigendirection = return_sloppiest_eigendirection(g_final)
indices = np.arange(len(final_sloppiest_eigendirection))
plt.figure(figsize=(10, 5))
plt.bar(indices, final_sloppiest_eigendirection, color='blue')
plt.axhline(0, color='gray', linewidth=0.5)  
plt.title("Sloppiest Eigendirection at Boundary", fontsize=25, color='black', backgroundcolor='white', pad=20)
plt.xticks(indices, labels, rotation=45, ha='right')  #
plt.xlabel("Bare Parameter", fontsize=18)
plt.ylabel("Bare Parameter Component", fontsize=18)
plt.tick_params(axis='both', labelsize=16)
plt.gca().set_facecolor("white") 
plt.gcf().patch.set_facecolor("white")  

# Display the plot
plt.tight_layout() 
plt.show()


"""
first run of MBAM shows we hit boundary gamma->0
I calculate the analytical limit of the model using Mathematica
"""
N_new = N-1


def r_new(y):
    λ, γo, F = np.exp(y)
    Ta = F/λ*(1-np.exp(-t*λ))
    To = F/λ*(γo*(1-np.exp(-t*λ))-λ*(1-np.exp(-t*γo)))/(γo-λ)
    return np.hstack((Ta, To))


def objective(y):
    predicted = r_new(y)
    msl = np.mean((predicted - noisy_data_point) ** 2)
    return msl

def RMS_rel_differences(new_predictions, original_data):
    return np.sqrt(np.mean((2*(new_predictions-original_data)/(new_predictions+original_data))**2))

def RRMSE(new_predictions, original_data):
    return np.sqrt(np.mean((new_predictions - original_data) ** 2))/np.linalg.norm(original_data)

def j_new(y):
    jacob = jacobian_func(r_new, M, N_new)
    return jacob(y)


def Avv_new(y, v_new):
    avv_function = Avv_func(r_new)
    return avv_function(y, v_new)


# Choose starting parameters
y0 = [-1.23981994, -4.76168556,  0.25466911]
y_result = minimize(objective, y0, method='L-BFGS-B')
print(y_result)
y = y_result.x  # new parameters obtained by fitting to original model predictions
print("Fit Error (Objective Function Value):")
print(y_result.fun)
print("RRMSE is:")
print(RRMSE(r_new(y), noisy_data_point))
print("RMS is:")
print(RMS_rel_differences(r_new(y), noisy_data_point))
print(np.linalg.norm(noisy_data_point))
v_new = initial_velocity(y, j_new, Avv_new)


def callback_new(g):
    _, s, _ = np.linalg.svd(j_new(g.xs[-1]))
    print(
        "Iteration: %i, tau: %f, |v| = %f, eigenvalue: %.25f"
        % (len(g.vs), g.ts[-1], np.linalg.norm(g.vs[-1]), s[-1])
    )
    return np.linalg.norm(g.vs[-1]) < 349


# Construct the geodesic
geo_new = Geodesic(r_new, j_new, Avv_new, y, v_new, atol=1e-2, rtol=1e-2,
                   parameterspacenorm=False, callback=callback_new)

# Integrate
geo_new.integrate(480)
plot_geodesic_path(geo_new, [colors[0]] + colors[2:],
                   [labels[0]]+labels[2:], N_new)

"""
second run of MBAM shows we hit boundary F, lambda go to infinity while their ratio remains finite
I calculate the analytical limit of the model using Mathematica
"""
N_new_new = N-2

def r_new_new(z):
    γo, c = np.exp(z)
    Ta = np.full(M//2, c)
    To = c*(1-np.exp(t*γo))
    return np.hstack((Ta, To))

def objective_new_new(z):
    predicted = r_new_new(z)
    msl = np.mean((predicted - noisy_data_point) ** 2)
    return msl

def j_new_new(z):
    jacob = jacobian_func(r_new_new, M, N_new_new)
    return jacob(z)


def Avv_new_new(z, v_new_new):
    avv_function = Avv_func(r_new_new)
    return avv_function(z, v_new_new)


# Choose starting parameters
z0 = [-4.76168556,  4.4]
z_result= minimize(objective_new_new, z0, method='L-BFGS-B')
print(z_result)
z = z_result.x  # new parameters obtained by fitting to original model predictions
print("Fit Error (Objective Function Value):")
print(z_result.fun)
print("RRMSE is:")
print(RRMSE(r_new_new(z), noisy_data_point))
v_new_new = initial_velocity(z, j_new_new, Avv_new_new)


def callback_new_new(g):
    _, s, _ = np.linalg.svd(j_new_new(g.xs[-1]))
    print(
        "Iteration: %i, tau: %f, |v| = %f, eigenvalue: %.25f"
        % (len(g.vs), g.ts[-1], np.linalg.norm(g.vs[-1]), s[-1])
    )
    return np.linalg.norm(g.vs[-1]) < 210


# Construct the geodesic
geo_new_new = Geodesic(r_new_new, j_new_new, Avv_new_new, z, v_new_new, atol=1e-2, rtol=1e-2,
                   parameterspacenorm=False, callback=callback_new_new)

# Integrate
geo_new_new.integrate(480)
plot_geodesic_path(geo_new_new, colors[2:],
                   [labels[2]]+['c'], N_new_new)
"""
third and last run of MBAM shows we hit boundary gamma_o -> 0
I calculate the analytical limit of the model using Mathematica
"""
def r_new_new_new(c):
    Ta=np.full(M//2, c)
    To=np.full(M//2, 0.)
    return np.hstack((Ta, To))


def objective_new_new_new(c):
    predicted = r_new_new_new(c)
    msl = np.mean((predicted - noisy_data_point) ** 2)
    return msl

result_c=minimize(objective_new_new_new, 4., method='L-BFGS-B')
final_c=result_c.x
print("Fit Error (Objective Function Value):")
print(result_c.fun)
print("RRMSE is:")
print(RRMSE(r_new_new_new(final_c), noisy_data_point))
print(final_c)
print("RMS is:")
print(RMS_rel_differences(r_new_new_new(final_c), noisy_data_point))


