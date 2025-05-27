"""
Solution to exercise 01: Linear regression and TDD

DISCLAIMER:
Please note that this solution may contain errors (please report them, thanks!),
and that there are most-likely more elegant and more efficient implementations available
for the given problem. In this light, this solution may only serve as an inspiration for
solving the given exercise tasks.

(C) Merten Stender, TU Berlin
merten.stender@tu-berlin.de
"""

from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt

"""
--- LINEAR REGRESSION: SCALAR LEAST-SQUARES Normal Equation
"""

from my_lin_regressor import lin_regressor


# simple test case: generate pseudo-linear data and regress
x = np.arange(0, 100, 1)
y = 2 + x + np.random.randn(len(x))
theta = lin_regressor(x=x, y=y)
theta_0, theta_1 = theta[0], theta[1]
# compute the predictions of the lin. regressor
y_hat = theta_0 + theta_1 * x

plt.figure()
plt.scatter(x, y)
plt.plot(x, y_hat, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['data', 'lin. fit'])
plt.title(rf'$\theta_0=${theta_0:.3f}, $\theta_1=${theta_1:.3f}')
plt.show()

# regression evaluation plot (y_pred vs. y_true)
plt.figure()
plt.plot(x, x, color='gray', label='perfect fit')
plt.scatter(y, y_hat, color='black', label='lin. regressor')
plt.legend()
plt.xlabel(r'ground truth $y$')
plt.ylabel(r'model prediction $\hat{y}$')
plt.show()


"""
--- DRIVING DATA CASE STUDY ---
"""
data = np.genfromtxt("driving_data.csv", delimiter=",")

# extract data
velocity = data[:, 0]   # car velocity [m/s]
power = data[:, 1]      # engine power [W]

# some constants as given in the exercise sheet
CW, A, RHO, G, M = 0.4, 1.5, 1.2, 9.81, 2400  # -, m^2, kg/m^3, m/s^2, kg

plt.figure()
plt.plot(velocity * 3.6, power / 1000, linestyle='none', marker='.', markersize=6)
plt.xlabel(r'driving velocity $v$ [km/h]')
plt.ylabel(r'Engine power $P_{\mathrm{engine}}$ [kW]')
plt.show()

"""
Step 0: Analytical considerations. Engine power must compensate wind and rolling resistance
to maintain the speed of the car.

P_engine = P_wind + P_roll
P_engine = v*cW*A*(rho*v**2)/(2) + v*cR*M*g*cos(alpha)

-> all coefficients but cR given; data given for variables P_engine and v

-> we want to have an equation of form a = cR*x, such that we can fit a lin. regressor
and interpret the coefficients of the model. 

P_engine - v*cW*A*(rho*v**2)/(2) = cR * (v*M*g*cos(alpha))
  
0 = theta_0 + theta_1*x
 
"""

"""
Step 1: compute the wind resistance
- assumption: driving velocity == relative velocity (no wind)
"""


def wind_force(v:float | np.ndarray)->float | np.ndarray:
    return CW * A * (RHO * v ** 2) / 2

# plot the rolling resistance power
power_roll = power - velocity * wind_force(v=velocity)

plt.figure()
plt.plot(velocity * 3.6, power_roll / 1000, linestyle='none', marker='.', markersize=6)
plt.xlabel(r'driving velocity $v$ [km/h]')
plt.ylabel('power [kW]')
plt.legend(['rolling resistance power'])
plt.show()

"""
Step 2: obtain rolling resistance force

rolling force must be the engine power without wind energy, then divided by the velocity
f_roll*v = p_roll = power_eng - power_wind = power_eng - (v*f_wind)

set up linear regression model 

p_roll = v * cR * M * g * cos(alpha)

--> we want to find cR:
y = theta_0 + theta_1 * (v * M * g * cos(alpha))

alpha (inclination) is unknown -> set it to zero
theta_1 will be cR
theta_0 will be some offset of the power (additional users, heating, etc.)
"""
# we need to expand the dimensions of the arrays to make them matrices of shape (N,1)
# (makes them compatible with sklearn package)
y = np.expand_dims(power_roll, axis=1)
x = np.expand_dims(velocity * M * G, axis=1)

# re-use our simple implementation from Task 1
theta = lin_regressor(x=x, y=y)
theta_0, theta_1 = theta[0], theta[1]

print(f'\n Custom regressor results: ')
print(f'linear regression model: \ty = {theta_0:.4f} + {theta_1:.4f} * x')
print(f'rolling resistance \t\t\tcR = {theta_1:.4f}')

plt.figure()
plt.plot(x, y / 1000, linestyle='none', marker='.', markersize=6)
plt.plot(x, (theta_0 + x * theta_1) / 1000, color='red')
plt.xlabel(r'$v\cdot M \cdot g$')
plt.ylabel('Rolling power resistance [kW]')
plt.legend(['data', 'lin. fit'])
plt.show()


# ------ scikit-learn version ------
regressor = LinearRegression(fit_intercept=True)
regressor.fit(X=x, y=y)

# get the model
theta_0 = float(regressor.intercept_)
theta_1 = float(regressor.coef_)

print(f'\n Scikit-Learn results: ')
print(f'linear regression model: \ty = {theta_0:.4f} + {theta_1:.4f} * x')
print(f'rolling resistance \t\t\tcR = {theta_1:.4f}')
# ------ ------ ------ ------ ------

# evaluate the R2 value
R2_score = regressor.score(X=x, y=y)
print(f'\nR2 score is {R2_score:.2f}')

# for plotting the linear regression line: query the regressor at 0 and max. velocity
X_eval = np.expand_dims(np.array([0, np.max(velocity * M * G)]), axis=1)
y_predict = regressor.predict(X=X_eval)

plt.figure()
plt.plot(velocity * 3.6, power_roll / 1000,
         linestyle='none', marker='.', markersize=6)
plt.plot(X_eval / M / G * 3.6, y_predict / 1000, color='red', linewidth=4)
plt.xlabel('driving velocity [km/h]')
plt.ylabel('Power [kW]')
plt.legend(['data', 'linear fit'])
plt.show()
