import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm, expm
import invariants_py.kinematics.orientation_kinematics as SO3
import time

# Generate a random skew-symmetric matrix
omega = SO3.crossmat(np.random.rand(3))
#omega = SO3.crossmat(np.array([np.pi, 0, 0])) # -- scipy logm fails for this case 
print("Skew-symmetric matrix omega:")
print(omega)

# Calculate the exponential of the skew-symmetric matrix using scipy and the custom method
start_time_scipy_expm = time.time()
exp_R_scipy = expm(omega)
end_time_scipy_expm = time.time()
scipy_expm_duration = end_time_scipy_expm - start_time_scipy_expm

start_time_custom_expm = time.time()
exp_R_custom = SO3.expm(omega)
end_time_custom_expm = time.time()
custom_expm_duration = end_time_custom_expm - start_time_custom_expm

print("Scipy expm:")
print(exp_R_scipy)
print(f"Scipy expm duration: {scipy_expm_duration} seconds")
print("Custom expm:")
print(exp_R_custom)
print(f"Custom expm duration: {custom_expm_duration} seconds")

# Calculate the difference between the two exponentials
diff_exp = exp_R_scipy - exp_R_custom
frobenius_norm_exp = np.linalg.norm(diff_exp, 'fro')
print("Difference between expm results:")
print(diff_exp)
print(f"Frobenius norm of expm difference: {frobenius_norm_exp}")

# Calculate the logarithm of the rotation matrix using scipy and the custom method
start_time_scipy_logm = time.time()
log_R_scipy = logm(exp_R_scipy)
end_time_scipy_logm = time.time()
scipy_logm_duration = end_time_scipy_logm - start_time_scipy_logm

start_time_custom_logm = time.time()
log_R_custom = SO3.logm(exp_R_custom)
end_time_custom_logm = time.time()
custom_logm_duration = end_time_custom_logm - start_time_custom_logm

print("Scipy logm:")
print(log_R_scipy)
print(f"Scipy logm duration: {scipy_logm_duration} seconds")
print("Custom logm:")
print(log_R_custom)
print(f"Custom logm duration: {custom_logm_duration} seconds")

# Calculate the difference between the two logarithms
diff_log = log_R_scipy - log_R_custom
frobenius_norm_log = np.linalg.norm(diff_log, 'fro')
print("Difference between logm results:")
print(diff_log)
print(f"Frobenius norm of logm difference: {frobenius_norm_log}")

# Plot the differences
plt.figure()
plt.plot(diff_exp.flatten())
plt.xlabel('Index')
plt.ylabel('Difference')
plt.title('Difference between the two expm results')
plt.show()

plt.figure()
plt.plot(diff_log.flatten())
plt.xlabel('Index')
plt.ylabel('Difference')
plt.title('Difference between the two logm results')
plt.show()
