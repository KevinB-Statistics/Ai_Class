import numpy as np
import matplotlib.pyplot as plt

#10 iterations
t_start = 0
t_end = 500
n_steps = 50

dt = (t_end - t_start)/n_steps
time_values = [t_start + i*dt for i in range(n_steps)]
#constant target setpoint = 10
setpoint_values = [10 for _ in time_values]
# print(time_values)
# print(setpoint_values)

measurements = [0]
errors = [setpoint_values[0] - measurements[0]]

#Proportional component
Kp = 0.5
#Integral component
Ki = 0.1
#Derivative component
Kd = 0.2

integral_values = [0]
proportional_values = [0]
derivative_values = [0]
pid_values = [0]
integral = 0

#Variable to store previous error value
previous_error = 0
calculation_steps = []

#Loop
for i in range(1, n_steps):
    e = errors[-1]
    integral += e*dt
    proportional = Kp*e
    derivative = Kd*(e-previous_error)/dt
    integral_contribution = Ki*integral
    pid = proportional + integral_contribution + derivative
    #Add the PID output to our current values
    measurements.append(measurements[-1] + pid)
    #Calculate error for the next step
    error = setpoint_values[-1] - measurements[-1]
    errors.append(error)
    integral_values.append(integral_contribution)
    proportional_values.append(proportional)
    derivative_values.append(derivative)
    pid_values.append(pid)

#Plot everything
plt.figure(figsize=(10,6))
plt.plot(time_values, measurements, label="Measurements")
plt.plot(time_values, pid_values, label="PID output")
plt.plot(time_values, proportional_values, label="Proportional")
plt.plot(time_values, integral_values, label="Integral")
plt.plot(time_values, derivative_values, label="Derivative")
plt.xlim(0, 100)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Change in PID output over time")
plt.grid(True)
plt.show()