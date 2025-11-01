import numpy as np
from get_distance_piecewise import solve_distance

# --- 1) Your data for the first 5 options ---

stock_price = np.array([437.11, 117.93, 224.31, 183.13, 476.79, 177.66, 434.47, 239.20, 209.78, 565.33])
strike_price = np.array([435, 117, 220, 180, 480, 175, 440, 235, 210, 560])
option_price = np.array([50.90, 19.71, 28.37, 28.35, 61.81, 24.85, 22.55, 57.10, 14.78, 64.45])
is_call = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=bool)

n = stock_price.size  # = 5

# --- 2) Build A, B, a, b for payoff y_i = max(A_i x + a_i, B_i x + b_i) ---

# Here x ∈ R^5 is the future stock‐price vector.
# For a CALL: payoff = max( x_i - K, 0 ) 
#    ⇒ A_i = e_iᵀ, a_i = -K_i ;  B_i = 0, b_i = 0
# For a PUT : payoff = max( K - x_i, 0 ) 
#    ⇒ A_i = -e_iᵀ, a_i =  K_i ;  B_i = 0, b_i = 0

A = np.zeros((n, n))
B = np.zeros((n, n))
a = np.zeros(n)
b = np.zeros(n)

for i in range(n):
    if is_call[i]:
        A[i, i] =  1.0
        a[i]    = -strike_price[i]
    else:
        A[i, i] = -1.0
        a[i]    =  strike_price[i]
    # B[i,:] stays all zeros, b[i] = 0

# --- 3) Define weights w and loss‐threshold nu ---
w  = np.array([0.2, 0.2, 0.2, 0.2, 0.2])   # e.g. equal weights
v = 2                                  # “I consider a loss of 20 units unacceptable”

# --- 4) Reference point x0 is today’s spot vector ---
x0 = stock_price.copy()

# --- 5) Call solve_distance, returning both x_star and distance ---
x_star, dist = solve_distance(
    x0=x0,
    A=A, B=B, a=a, b=b,
    w=w, v=v,
    return_x_star=True
)

print("Closest stress‐scenario x*       =", x_star)
print("Euclidean distance to breach     =", dist)
