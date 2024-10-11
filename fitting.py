from scipy.optimize import curve_fit
import numpy as np

def recalibrate_parameters(t, Ta_actual, To_actual, initial_params, F_fixed=1e4):
    """
    Recalibrates the parameters λ, γ, and γ0 by fixing F to a constant value (F_fixed)
    and using least squares fitting on the given data points for temperature anomalies (Ta, To).

    Parameters:
    - t: array of time points
    - Ta_actual: array of actual temperature anomalies for the atmosphere
    - To_actual: array of actual temperature anomalies for the ocean
    - initial_params: initial guesses for the other parameters [λ, γ, γ0]
    - F_fixed: fixed value for the forcing parameter F (default: 1e4)

    Returns:
    - opt_params: recalibrated parameters [λ, γ, γ0]
    """
    
    # Define the model with fixed F and recalibrating λ, γ, γ0
    def model(t, λ, γ, γ0):
        # Reuse your existing function for r_new with fixed F
        b = λ + γ + γ0
        b_star = λ + γ - γ0
        delta = b * b - 4 * λ * γ0

        # Mode parameters (Fast and Slow)
        τ_f = (b - np.sqrt(delta)) / (2 * γ0 * λ)
        τ_s = (b + np.sqrt(delta)) / (2 * γ0 * λ)

        φ_s = 1 / (2 * γ) * (b_star + np.sqrt(delta))
        φ_f = 1 / (2 * γ) * (b_star - np.sqrt(delta))

        a_f = φ_s * τ_f * λ / (φ_s - φ_f)
        a_s = -φ_f * τ_s * λ / (φ_s - φ_f)

        # Temperature anomalies
        Ta = F_fixed / λ * (a_f * (1 - np.exp(-1 / τ_f * t)) + a_s * (1 - np.exp(-1 / τ_s * t)))
        To = F_fixed / λ * (a_f * φ_f * (1 - np.exp(-1 / τ_f * t)) + φ_s * a_s * (1 - np.exp(-1 / τ_s * t)))
        
        return np.concatenate((Ta, To))

    # Objective function for least squares fitting (combining Ta and To errors)
    def residuals(params, t, Ta_actual, To_actual):
        λ, γ, γ0 = params
        Ta_pred, To_pred = np.split(model(t, λ, γ, γ0), 2)
        # Compute residuals (difference between actual and predicted values)
        residuals_Ta = Ta_actual - Ta_pred
        residuals_To = To_actual - To_pred
        return np.concatenate((residuals_Ta, residuals_To))  # Combine both residuals into a single array

    # Perform the least squares fitting using curve_fit
    opt_params, _ = curve_fit(lambda t, λ, γ, γ0: model(t, λ, γ, γ0), t, np.concatenate((Ta_actual, To_actual)),
                              p0=initial_params)

    return opt_params


from scipy.optimize import curve_fit

def recalibrate_gamma0(t, Ta_actual, To_actual, lambda_fixed, gamma_fixed, initial_gamma0, F_fixed=1e4):
    """
    Recalibrates the parameter γ0 by fixing F, λ, and γ to constant values and 
    using least squares fitting on the given data points for temperature anomalies (Ta, To).

    Parameters:
    - t: array of time points
    - Ta_actual: array of actual temperature anomalies for the atmosphere
    - To_actual: array of actual temperature anomalies for the ocean
    - lambda_fixed: fixed value for λ (climate feedback parameter)
    - gamma_fixed: fixed value for γ (ocean-atmosphere heat exchange coefficient)
    - initial_gamma0: initial guess for γ0
    - F_fixed: fixed value for the forcing parameter F (default: 1e4)

    Returns:
    - opt_gamma0: recalibrated γ0
    """

    # Define the model with fixed F, λ, and γ, and recalibrating γ0
    def model(t, gamma0):
        # Reuse your existing function for r_new with fixed F, λ, and γ
        b = lambda_fixed + gamma_fixed + gamma0
        b_star = lambda_fixed + gamma_fixed - gamma0
        delta = b * b - 4 * lambda_fixed * gamma0

         # Avoid negative values in the square root
        if delta < 0:
            print(f"Warning: Negative delta encountered: {delta}")
            delta = 0

        # Mode parameters (Fast and Slow)
        τ_f = (b - np.sqrt(delta)) / (2 * gamma0 * lambda_fixed)
        τ_s = (b + np.sqrt(delta)) / (2 * gamma0 * lambda_fixed)

        φ_s = 1 / (2 * gamma_fixed) * (b_star + np.sqrt(delta))
        φ_f = 1 / (2 * gamma_fixed) * (b_star - np.sqrt(delta))

        a_f = φ_s * τ_f * lambda_fixed / (φ_s - φ_f)
        a_s = -φ_f * τ_s * lambda_fixed / (φ_s - φ_f)

         # Check for any infinities or NaNs
        if any(np.isnan([a_f, a_s, τ_f, τ_s])) or any(np.isinf([a_f, a_s, τ_f, τ_s])):
            print(f"Warning: NaN or Inf encountered in parameters a_f, a_s, tau_f, tau_s")

        # Temperature anomalies
        Ta = F_fixed / lambda_fixed * (a_f * (1 - np.exp(-1 / τ_f * t)) + a_s * (1 - np.exp(-1 / τ_s * t)))
        To = F_fixed / lambda_fixed * (a_f * φ_f * (1 - np.exp(-1 / τ_f * t)) + φ_s * a_s * (1 - np.exp(-1 / τ_s * t)))

        if np.any(np.isnan(Ta)) or np.any(np.isnan(To)) or np.any(np.isinf(Ta)) or np.any(np.isinf(To)):
            print(f"Warning: NaN or Inf encountered in Ta or To")
        
        return np.concatenate((Ta, To))

    # Objective function for least squares fitting (combining Ta and To errors)
    def residuals(gamma0, t, Ta_actual, To_actual):
        Ta_pred, To_pred = np.split(model(t, gamma0), 2)
        # Compute residuals (difference between actual and predicted values)
        residuals_Ta = Ta_actual - Ta_pred
        residuals_To = To_actual - To_pred
        return np.concatenate((residuals_Ta, residuals_To))  # Combine both residuals into a single array

    # Perform the least squares fitting using curve_fit, optimizing only γ0
    opt_gamma0, _ = curve_fit(lambda t, gamma0: model(t, gamma0), t, np.concatenate((Ta_actual, To_actual)),
                              p0=[initial_gamma0], bounds=(0, np.inf), maxfev=3000)

    return opt_gamma0[0]

