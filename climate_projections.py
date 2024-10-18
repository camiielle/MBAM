import numpy as np
import matplotlib.pyplot as plt

# Function to compute climate model
def model(x, t):
    λ, γ, γ0, F = x

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

    return Ta, To


def Ta(x, t):
    return model(x, t)[0]

def To(x, t):
    return model(x, t)[1]

def diff(x,t):
    return Ta(x,t)-To(x,t)


# Define the parameter ranges for λ, γ, γ0, F and the time points
times = [0, 10, 100, 500]  # years
parameter_ranges = [np.linspace(0.0001, 10, 40), np.linspace(0.1, 100, 40), np.linspace(
    0.01, 1, 40), np.linspace(-10, 100, 40)]  # λ, γ, γ0, F
x0 = [1.3/8, 0.0875, 0.007, 0.4875]  # λ, γ, γ0, F initial values

colors = ['#FF5733', '#33C1FF', '#33FF57', '#FFFF33']
labels = ['λ', 'γ', 'γ0', 'F']


# Function to create individual 3D projection
def create_3D_projection(ax, param1_index, param2_index, z_label, function):
    x_vals = parameter_ranges[param1_index]
    y_vals = parameter_ranges[param2_index]
    X, Y = np.meshgrid(x_vals, y_vals)

    for time_idx, t in enumerate(times):
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                params = x0
                params[param1_index] = x_vals[i]
                params[param2_index] = y_vals[j]
                Z[i, j] = function(params, t)

        # Plot the surface with a specific color, subtle grid, and no edge lines
        ax.plot_surface(
            X, Y, Z, color=colors[time_idx], alpha=0.5, edgecolor='gray', linewidth=0.6)

    ax.set_xlabel(labels[param1_index])
    ax.set_ylabel(labels[param2_index])
    ax.set_zlabel(z_label)
    ax.grid(True, linestyle=':', linewidth=0.5, color='gray')

# Create the triangular layout for the projections
def triangular_layout(function, label):
    fig = plt.figure(figsize=(8, 5))
    subplots = []
    for i in [1, 2, 3, 5, 6, 9]:
        subplots.append(fig.add_subplot(3, 3, i, projection='3d'))
    sublot_index = 0
    for param1_index in range(3):
        for param2_index in range(param1_index+1, 4):
            create_3D_projection(
                subplots[sublot_index], param1_index, param2_index, label, function)
            sublot_index += 1

    # Adjust spacing to bring the first row closer to the title and increase space between rows
    plt.subplots_adjust(wspace=0.3, hspace=2.)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()

# Generate the triangular layout for Ta and To
triangular_layout(Ta, 'Ta')
triangular_layout(To, 'To')
triangular_layout(diff, 'Ta-To')

# Generate the grid layout for Ta and To
# grid_layout('Ta')
# grid_layout('To')
