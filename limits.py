# evaluates limits of my model
import sympy as sp

F = sp.symbols('F')
t = sp.symbols('t', positive=True)
s = sp.symbols('s', positive=True)
γ0 = sp.symbols('γ0', positive=True)
γ = sp.symbols('γ', positive=True)
λ = sp.symbols('λ')

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

# limit_lambda = sp.limit(Ta, λ, -sp.oo)
# simultaneous_limit = sp.limit(Ta.subs({γ: s, λ: -s}), s, sp.oo)  # First take limit with respect to x
limit_F = sp.limit(Ta, F, sp.oo)  # First take limit with respect to x
print(limit_F)  # oo*sign(((1 - exp(-2*t*γ0*λ/(γ + γ0 + λ - sqrt(-4*γ0*λ + (γ + γ0 + λ)**2))))*(γ - γ0 + λ + sqrt(-4*γ0*λ + (γ + γ0 + λ)**2))*(γ + γ0 + λ - sqrt(-4*γ0*λ + (γ + γ0 + λ)**2)) - (1 - exp(-2*t*γ0*λ/(γ + γ0 + λ + sqrt(-4*γ0*λ + (γ + γ0 + λ)**2))))*(γ - γ0 + λ - sqrt(-4*γ0*λ + (γ + γ0 + λ)**2))*(γ + γ0 + λ + sqrt(-4*γ0*λ + (γ + γ0 + λ)**2)))/(λ*sqrt(-4*γ0*λ + (γ + γ0 + λ)**2)))

# print(simultaneous_limit) #oo*sign(F)

# limit lambda oo*sign(F)
# limit gamma 0
# double limit
