import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to compute Ta and To based on the parameter set
def compute_Ta_To(params, t):
    λ, γ, γ0, F = params
    
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

# Define the parameter ranges for λ, γ, γ0, F and the time points
lambda_vals = np.linspace(0.01, 2, 40)  # Generate 40 points between -2 and 2
#lambda_vals_filtered = lambda_vals[(np.abs(lambda_vals) > 0.01) & (np.abs(lambda_vals + 1) > 0.3)]
# lambda_vals = lambda_vals[np.abs(lambda_vals) > 0.1]  # Exclude values near zero (e.g., abs < 0.1)
gamma_vals = np.linspace(0.1, 20, 40)
gamma0_vals = np.linspace(0.01, 1, 40)
F_vals = np.linspace(0.1, 20, 40)
times = [0, 10, 100, 500]  # years

# Define a list of distinct colors for each surface corresponding to different time points
colors = ['#FF5733', '#33C1FF', '#33FF57', '#FFFF33'] 

# Function to create individual 3D projection
def create_3D_projection(ax, x_vals, y_vals, param1_name, param2_name, z_label, Ta_or_To):
    X, Y = np.meshgrid(x_vals, y_vals)
    
    for time_idx, t in enumerate(times):
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                params = [lambda_vals[i], gamma_vals[j], gamma0_vals[j], F_vals[j]]
                Ta, To = compute_Ta_To(params, t)
                Z[i, j] = Ta if Ta_or_To == "Ta" else To
        
        # Plot the surface with a specific color, subtle grid, and no edge lines
        ax.plot_surface(X, Y, Z, color=colors[time_idx], alpha=0.5, edgecolor='gray', linewidth=0.6)

    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_zlabel(z_label)
    ax.grid(True, linestyle=':', linewidth=0.5, color='gray')

# Create the triangular layout for the projections
def triangular_layout(Ta_or_To):
    fig = plt.figure(figsize=(8, 5))

    projections = [
        ('λ', lambda_vals, 'γ', gamma_vals),
        ('λ', lambda_vals, 'γ0', gamma0_vals),
        ('λ', lambda_vals, 'F', F_vals),
        ('γ', gamma_vals, 'γ0', gamma0_vals),
        ('γ', gamma_vals, 'F', F_vals),
        ('γ0', gamma0_vals, 'F', F_vals),
    ]

    # Top row (three images)
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    create_3D_projection(ax1, lambda_vals, gamma_vals, 'λ', 'γ', Ta_or_To, Ta_or_To)

    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    create_3D_projection(ax2, lambda_vals, gamma0_vals, 'λ', 'γ0', Ta_or_To, Ta_or_To)

    ax3 = fig.add_subplot(3, 3, 3, projection='3d')
    create_3D_projection(ax3, lambda_vals, F_vals, 'λ', 'F', Ta_or_To, Ta_or_To)

    # Middle row (two images)
    ax4 = fig.add_subplot(3, 3, 5, projection='3d')
    create_3D_projection(ax4, gamma_vals, gamma0_vals, 'γ', 'γ0', Ta_or_To, Ta_or_To)

    ax5 = fig.add_subplot(3, 3, 6, projection='3d')
    create_3D_projection(ax5, gamma_vals, F_vals, 'γ', 'F', Ta_or_To, Ta_or_To)

    # Bottom row (one image)
    ax6 = fig.add_subplot(3, 3, 9, projection='3d')
    create_3D_projection(ax6, gamma0_vals, F_vals, 'γ0', 'F', Ta_or_To, Ta_or_To)

    # Adjust spacing to bring the first row closer to the title and increase space between rows
    plt.subplots_adjust(wspace=0.3, hspace=2.)  # Closer title, more space between rows

    plt.tight_layout(rect=[0, 0, 1, 1])  # Bring the first row closer to the title
    plt.show()

def grid_layout(Ta_or_To):
    fig = plt.figure(figsize=(12, 8))  # Adjust figure size for 2 rows of 3 columns

    projections = [
        ('λ', lambda_vals, 'γ', gamma_vals),
        ('λ', lambda_vals, 'γ0', gamma0_vals),
        ('λ', lambda_vals, 'F', F_vals),
        ('γ', gamma_vals, 'γ0', gamma0_vals),
        ('γ', gamma_vals, 'F', F_vals),
        ('γ0', gamma0_vals, 'F', F_vals),
    ]

    # Create the 2x3 grid layout (two rows, three columns)
    for idx, (x_label, x_vals, y_label, y_vals) in enumerate(projections):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        create_3D_projection(ax, x_vals, y_vals, x_label, y_label, Ta_or_To, Ta_or_To)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.6)  # Adjust these values as needed
    plt.suptitle(f'Grid Projection Layout for {Ta_or_To}', fontsize=12, y=0.98)  # Title for the layout
    plt.tight_layout(rect=[0, 0, 1, 1])  # Bring first row closer to title, adjust overall layout
    plt.show()

# Generate the triangular layout for Ta and To
triangular_layout('Ta')
triangular_layout('To')

# Generate the grid layout for Ta and To
grid_layout('Ta')
grid_layout('To')