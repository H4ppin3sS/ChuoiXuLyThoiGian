import pandas as pd
import numpy as np

df = pd.read_csv('PFE_US_Equity.csv') 

# Chuyển đổi ngày và log giá
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['log_price'] = np.log(df['PX_LAST'])

observations = df['log_price'].values
T = len(observations)

# Khởi tạo các giá trị
num_particles = 100
num_iterations = 10  
Q = 1e-4  
R = 1e-3  

def particle_filter(y, Q, R, num_particles):
    T = len(y)
    particles = np.zeros((T, num_particles))
    weights = np.zeros((T, num_particles))

    particles[0] = np.random.normal(loc=y[0], scale=np.sqrt(R), size=num_particles)
    weights[0] = np.ones(num_particles) / num_particles

    for t in range(1, T):
        particles[t] = np.random.normal(loc=particles[t-1], scale=np.sqrt(Q))
        
        weights[t] = (
            1 / np.sqrt(2 * np.pi * R) * 
            np.exp(-0.5 * ((y[t] - particles[t]) ** 2) / R)
        )
        weights[t] /= np.sum(weights[t])  
        
        indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights[t])
        particles[t] = particles[t][indices]
        weights[t] = np.ones(num_particles) / num_particles

    return particles, weights

# EM iterations
Q_list, R_list = [Q], [R]
for iteration in range(num_iterations):
    # E-step
    particles, weights = particle_filter(observations, Q, R, num_particles)
    # x_hat = np.average(particles, axis=1, weights=weights[-1])
    x_hat = np.sum(particles * weights, axis=1)

    # M-step
    Q = np.mean((x_hat[1:] - x_hat[:-1]) ** 2)
    R = np.mean((observations - x_hat) ** 2)
    Q_list.append(Q)
    R_list.append(R)

print('Q Final: ', Q )
print('R Final: ', R )
print('Q_list: ', Q_list)
print('R_list: ', R_list)
# print(df)
print('Particles: ', particles)
print('Weights: ', weights)

# import matplotlib.pyplot as plt

# plt.plot(Q_list, label='Q')
# plt.plot(R_list, label='R')
# plt.xlabel('Iteration')
# plt.ylabel('Variance')
# plt.title('EM convergence of Q and R')
# plt.legend()
# plt.grid()
# plt.show()
