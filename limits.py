# evaluates limits of my model
import sympy as sp

F = sp.symbols('F', positive=True)
t = sp.symbols('t', positive=True)
γ0 = sp.symbols('γ0', positive=True)
γ = sp.symbols('γ', positive=True)
λ = sp.symbols('λ', positive=True)

# General parameters
b = λ + γ + γ0
b_star = λ + γ - γ0
delta = b*b - 4 * λ * γ0

# Mode parameters (Fast and Slow)
τ_f = (b - sp.sqrt(delta)) / (2 * γ0 * λ)
τ_s = (b + sp.sqrt(delta)) / (2 * γ0 * λ)

φ_s = 1 / (2 * γ) * (b_star + sp.sqrt(delta))
φ_f = 1 / (2 * γ) * (b_star - sp.sqrt(delta))

a_f = φ_s * τ_f * λ / (φ_s - φ_f)
l_f = a_f * τ_f * λ / (1 + γ/γ0)

a_s = -φ_f * τ_s * λ / (φ_s - φ_f)
l_s = a_s * τ_s * λ / (1 + γ/γ0)

# standard deviations
sigma_a = 0.1
sigma_o = 0.1

# defining ODEs solutions
Ta = F/λ*(a_f*(1-sp.exp(-t/τ_f)) + a_s*(1-sp.exp(-t/τ_s)))
To = F/λ*(a_f*φ_f*(1-sp.exp(-t/τ_f)) + φ_s*a_s*(1-sp.exp(-t/τ_s)))

limit = sp.limit(Ta, γ, 0)  # First take limit with respect to x
print(limit)  