# ============================================================
# Section 1: introduce the model of a pendulum
# ============================================================

# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt

# # Parameters
# M = 1.0     # Mass of the cart (kg)
# m = 0.1     # Mass of the pendulum (kg)
# l = 1.0     # Length of the pendulum (m)
# g = 9.81    # Gravitational acceleration (m/s^2)

# # Input force (horizontal force on the cart)
# def input_force(t):
#     return 2.0 * np.sin(0.5 * t)  # Example: sinusoidal input

# # Equations of motion
# def cart_pendulum_dynamics(t, y):
#     x, x_dot, theta, theta_dot = y
    
#     # Horizontal force
#     F = input_force(t)
    
#     # Nonlinear equations of motion
#     sin_theta = np.sin(theta)
#     cos_theta = np.cos(theta)
#     denom = M + m * (1 - cos_theta**2)
    
#     x_ddot = (F + m * sin_theta * (l * theta_dot**2 + g * cos_theta)) / denom
#     theta_ddot = (-F * cos_theta - m * l * theta_dot**2 * cos_theta * sin_theta - (M + m) * g * sin_theta) / (l * denom)
    
#     return [x_dot, x_ddot, theta_dot, theta_ddot]

# # Initial conditions: [x, x_dot, theta, theta_dot]
# y0 = [0.0, 0.0, np.pi / 6, 0.0]  # Start with pendulum at 30 degrees

# # Time span
# t_span = (0, 10)  # Simulate for 10 seconds
# t_eval = np.linspace(*t_span, 1000)  # Time points for solution

# # Solve the system using solve_ivp
# sol = solve_ivp(cart_pendulum_dynamics, t_span, y0, t_eval=t_eval, method='RK45')

# # Extract results
# t = sol.t
# x, x_dot, theta, theta_dot = sol.y

# # Plot results
# plt.figure(figsize=(12, 8))

# # Input force
# plt.subplot(3, 1, 1)
# plt.plot(t, [input_force(ti) for ti in t], label="Input Force (F)")
# plt.xlabel("Time (s)")
# plt.ylabel("Force (N)")
# plt.title("Input Force vs Time")
# plt.legend()
# plt.grid()

# # Cart position
# plt.subplot(3, 1, 2)
# plt.plot(t, x, label="Cart Position (x)")
# plt.xlabel("Time (s)")
# plt.ylabel("Position (m)")
# plt.title("Cart Position vs Time")
# plt.legend()
# plt.grid()

# # Pendulum angle
# plt.subplot(3, 1, 3)
# plt.plot(t, theta, label="Pendulum Angle (θ)")
# plt.xlabel("Time (s)")
# plt.ylabel("Angle (rad)")
# plt.title("Pendulum Angle vs Time")
# plt.legend()
# plt.grid()

# plt.tight_layout()
# plt.show()

# ============================================================
# Section 2: generate training and testing data from the model
# ============================================================

# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# # System parameters
# M = 1.0     # Mass of the cart (kg)
# m = 0.1     # Mass of the pendulum (kg)
# l = 1.0     # Length of the pendulum (m)
# g = 9.81    # Gravitational acceleration (m/s^2)

# # Input force
# def random_force(t):
#     return np.random.uniform(-5, 5)  # Random force between -5 and 5

# # Cart-pendulum dynamics
# def cart_pendulum_dynamics(t, y, F):
#     x, x_dot, theta, theta_dot = y
#     sin_theta = np.sin(theta)
#     cos_theta = np.cos(theta)
#     denom = M + m * (1 - cos_theta**2)

#     x_ddot = (F + m * sin_theta * (l * theta_dot**2 + g * cos_theta)) / denom
#     theta_ddot = (-F * cos_theta - m * l * theta_dot**2 * cos_theta * sin_theta - (M + m) * g * sin_theta) / (l * denom)

#     return [x_dot, x_ddot, theta_dot, theta_ddot]

# # Generate data
# def generate_data(num_trajectories=100, time_span=(0, 5), num_points=100):
#     X, y = [], []
#     for _ in range(num_trajectories):
#         y0 = [np.random.uniform(-1, 1), 0, np.random.uniform(-np.pi/4, np.pi/4), 0]
#         t_eval = np.linspace(*time_span, num_points)
#         F = np.random.uniform(-5, 5, size=len(t_eval))  # Random forces
#         sol = solve_ivp(lambda t, y: cart_pendulum_dynamics(t, y, np.interp(t, t_eval, F)),
#                         time_span, y0, t_eval=t_eval)
#         states = sol.y.T  # [x, x_dot, theta, theta_dot]
#         derivatives = np.array([cart_pendulum_dynamics(t, y, F[i]) for i, (t, y) in enumerate(zip(t_eval, states))])

#         X.extend(np.hstack([states, F.reshape(-1, 1)]))  # Input: state + force
#         y.extend(derivatives)  # Output: derivatives

#     return np.array(X), np.array(y)

# # Generate training and testing data
# X, y = generate_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Neural network setup
# model = Sequential([
#     Dense(64, activation='relu', input_dim=X.shape[1]),
#     Dense(64, activation='relu'),
#     Dense(y.shape[1])  # Output layer matches the dimension of derivatives
# ])
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# # Train the neural network
# history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# # Test the model
# y_pred = model.predict(X_test)

# # Evaluate trajectories
# def simulate_nn_trajectory(y0, F, time_span, model, num_points=100):
#     t_eval = np.linspace(*time_span, num_points)
#     y_pred = [y0]
#     for i in range(len(t_eval) - 1):
#         dt = t_eval[i + 1] - t_eval[i]
#         state = np.append(y_pred[-1], F[i])
#         derivatives = model.predict(state.reshape(1, -1))[0]
#         y_pred.append(y_pred[-1] + derivatives * dt)
#     return np.array(y_pred), t_eval

# # Generate a random test trajectory
# y0 = [0.0, 0.0, np.pi / 6, 0.0]
# F = np.random.uniform(-5, 5, size=100)
# time_span = (0, 5)
# true_sol = solve_ivp(lambda t, y: cart_pendulum_dynamics(t, y, np.interp(t, np.linspace(*time_span, len(F)), F)),
#                      time_span, y0, t_eval=np.linspace(*time_span, 100))
# nn_sol, t_eval = simulate_nn_trajectory(y0, F, time_span, model)

# # Plot true vs predicted trajectories
# plt.figure(figsize=(12, 8))
# labels = ['x', 'x_dot', 'theta', 'theta_dot']
# for i in range(4):
#     plt.subplot(4, 1, i + 1)
#     plt.plot(true_sol.t, true_sol.y[i], label='True')
#     plt.plot(t_eval, nn_sol[:, i], '--', label='NN')
#     plt.ylabel(labels[i])
#     plt.legend()
#     plt.grid()
# plt.xlabel('Time (s)')
# plt.suptitle('True vs NN Predicted Trajectories')
# plt.tight_layout()
# plt.show()

# # Plot error
# error = np.abs(true_sol.y.T - nn_sol)
# plt.figure(figsize=(12, 6))
# for i, label in enumerate(labels):
#     plt.plot(t_eval, error[:, i], label=f'Error in {label}')
# plt.xlabel('Time (s)')
# plt.ylabel('Error')
# plt.legend()
# plt.title('Prediction Error')
# plt.grid()
# plt.show()

# ============================================================
# Section 3: Control using AI
# ============================================================


import numpy as np
from scipy.integrate import solve_ivp
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# System parameters
M = 1.0     # Mass of the cart (kg)
m = 0.1     # Mass of the pendulum (kg)
l = 1.0     # Length of the pendulum (m)
g = 9.81    # Gravitational acceleration (m/s^2)

# Cart-pendulum dynamics
def cart_pendulum_dynamics(t, y, F):
    x, x_dot, theta, theta_dot = y
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    denom = M + m * (1 - cos_theta**2)

    x_ddot = (F + m * sin_theta * (l * theta_dot**2 + g * cos_theta)) / denom
    theta_ddot = (-F * cos_theta - m * l * theta_dot**2 * cos_theta * sin_theta - (M + m) * g * sin_theta) / (l * denom)

    return [x_dot, x_ddot, theta_dot, theta_ddot]

# Initial condition: Pendulum is vertical down and cart at origin
y0 = [0.0, 0.0, np.pi, 0.0]  # [x, x_dot, theta, theta_dot]

# Time span
time_span = (0, 10)
t_eval = np.linspace(*time_span, 1000)

# Reinforcement Learning Setup
class DQNPendulumController:
    def __init__(self):
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def build_model(self):
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(5,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)  # Output is the force F
        ])
        return model

    def train(self, X, y, epochs=50, batch_size=32):
        dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)
        for epoch in range(epochs):
            for x_batch, y_batch in dataset:
                with tf.GradientTape() as tape:
                    predictions = self.model(x_batch, training=True)
                    loss = tf.reduce_mean(tf.square(predictions - y_batch))
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def predict(self, X):
        return self.model(X)

# Generate data
def generate_rl_data(num_samples=10000):
    X, y = [], []
    for _ in range(num_samples):
        y0 = [np.random.uniform(-1, 1), 0, np.pi, 0]  # Random initial conditions
        F = np.random.uniform(-10, 10)  # Random force
        sol = solve_ivp(cart_pendulum_dynamics, time_span, y0, args=(F,), t_eval=t_eval)
        derivatives = np.array([cart_pendulum_dynamics(t, y, F) for t, y in zip(sol.t, sol.y.T)])
        X.extend(np.hstack([sol.y.T, np.full((len(sol.t), 1), F)]))
        y.extend(derivatives)
    return np.array(X), np.array(y)

X_train, y_train = generate_rl_data()

# Train DQN Controller
controller = DQNPendulumController()
controller.train(X_train, y_train, epochs=100)

# Test the controller
F_test = controller.predict(X_train[:, :-1])
true_sol = solve_ivp(cart_pendulum_dynamics, time_span, y0, t_eval=t_eval)

# Simulate closed-loop system
def simulate_closed_loop(y0, controller, time_span, num_points=100):
    t_eval = np.linspace(*time_span, num_points)
    y_pred = [y0]
    for i in range(len(t_eval) - 1):
        state = np.append(y_pred[-1], F_test[i])
        derivatives = cart_pendulum_dynamics(t_eval[i], state, F_test[i])
        y_pred.append(y_pred[-1] + np.array(derivatives) * (t_eval[i + 1] - t_eval[i]))
    return np.array(y_pred), t_eval

rl_sol, t_eval = simulate_closed_loop(y0, controller, time_span)

# Plot results
plt.figure(figsize=(12, 8))
labels = ['x', 'x_dot', 'theta', 'theta_dot']
for i in range(4):
    plt.subplot(4, 1, i + 1)
    plt.plot(true_sol.t, true_sol.y[i], label='True')
    plt.plot(t_eval, rl_sol[:, i], '--', label='RL Control')
    plt.ylabel(labels[i])
    plt.legend()
    plt.grid()
plt.xlabel('Time (s)')
plt.suptitle('True vs RL Controller Stabilized Trajectories')
plt.tight_layout()
plt.show()


