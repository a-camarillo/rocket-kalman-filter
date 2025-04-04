"""
State Estimation for Rocket Trajectory
Author: Anthony Camarillo
"""
import control as ct
import numpy as np
import scipy
import matplotlib.pyplot as plt

"""
From equations of motion for vertical rocket trajectory, the state variables are given
as altitude: x(t), velocity: v(t) = x_dot(t), and mass m(t)

Equations of motion can then be given as
x(i+1) = x(i) + v(i)*dt # Newton's equation of motion
v(i+1) = v(i) + (-g - (1/2)*rho*v(i)*abs(v(i))*(Cd*A)/m(i) + v(i)/abs(v(i))*(m_fuel(i)*u(i))/m(i))*dt
m(i+1) = m(i) + (-m_fuel(i)*dt)
"""

# Setting some initial conditions

Cd = 0.5 # drag coefficient
A = 3.7 # m, frontal area of Falcon 9
rho = 1.293 # kg/m^3, density of air
g = 9.81 # kg/m^2, gravitational constant

m0 = 549054 # kg, initial mass of Falcon 9
m_dry = 25600 # kg, dry mass of Falcon 9

# mass flow rate
# mfr = 2555 # kg/s, approximate mass flow rate of Falcon 9

# exhaust velocity
u = 282*g # m/s, falcon 9 specific impulse times gravitational constant

def trajectory():
    dt = 0.01
    t = np.arange(0,1000, dt)

    x = t*0
    v = t*0
    m = t*0

    mfr = 2555

    x[0] = 20
    v[0] = 2
    m[0] = m0

    for i in range(len(t)-1):
        if m[i] > m_dry:
            x[i+1] = x[i] + v[i]*dt
            v[i+1] = v[i] + (-g - 0.5*rho*v[i]*np.abs(v[i])*(Cd*A)/m[i]+ v[i]/np.abs(v[i])*(mfr*u/m[i]))*dt
            m[i+1] = m[i] + (-mfr)*dt
        else:
            x[i+1] = x[i] + v[i]*dt
            v[i+1] = v[i] + (-g - 0.5*rho*v[i]*np.abs(v[i])*(Cd*A)/m[i])*dt
            m[i+1] = m_dry 
        

    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1.plot(t,x)
    ax2.plot(t,v)
    ax3.plot(t,m)
    ax1.set_ylabel('Position')
    ax2.set_ylabel('Velocity')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Mass')

    plt.show()

def kalman_1D():
    """
    For a 1-D Rocket Simulation
    states are position, x and velocity
    x_dot = v
    v_dot = acceleration
    """
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    C = np.array([0, 1])

    Q = 0.01 * np.eye(2) # process noise covariance
    R = np.eye(2) # initial covariance

    # initial state and covariance
    x_hat = np.array([[0], [0]])
    P = np.eye(2)

    dt = 0.1
    t = np.arange(0, 1000, dt)
    accel = g*np.ones(np.size(t))
    pos_est = np.zeros(len(t))

    for k in range(len(t)-1):
        # prediction step
        x_hat = A @ x_hat + B * accel[k]
        P = A @ P @ A.T + Q

        # measurement update (simulated noisy GPS)
        z = x_hat[0] + np.random.randint(0,10) * np.sqrt(R)
        K = P @ C.T / (C @ P @ C.T + R)
        x_hat = x_hat + K @ (z - C @ x_hat) 
        P = (np.eye(2) - K @ C) @ P

        pos_est[k] = x_hat[0][1]

    fig, ax1 = plt.subplots()
    ax1.plot(t,pos_est)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Estimated Position')
    plt.show()

def discrete_ekf():
    dt = 0.1
    t = np.arange(0,800,dt)

    x0, v0, m0 = (0.,1.,549054.)
    x_true = np.array([[x0],[v0],[m0]]) # true state
    x_est = x_true # estimated state 
    P = np.diag([100., 10., 1.]) # initial covariance

    # process and measurement noise
    Q = np.diag([10, 10, 10])
    R = np.diag([10, 5])

    x_store = np.zeros((3, len(t)))
    x_est_store = np.zeros((3, len(t)))

    mfr = 2555
    dm = 0

    for i in range(len(t)-1):
        if x_true[2][0] > m_dry:
            # updating position
            x_true[0][0] = x_true[0][0] + x_true[1][0]*dt + Q[0][0]*np.random.randn(1)[0]
            # updating velocity
            x_true[1][0] = x_true[1][0] + (-g - 0.5*rho*x_true[1][0]*np.abs(x_true[1][0])*(Cd*A)/x_true[2][0]+ x_true[1][0]/np.abs(x_true[1][0])*(mfr*u/x_true[2][0]))*dt + Q[1][1]*np.random.rand(1)[0]
            # updating mass
            x_true[2][0] = x_true[2][0] + (-mfr)*dt + Q[2][2]*np.random.randn(1)[0]
        else:
            # set dry mass conditions
            
            # updating position
            x_true[0][0] = x_true[0][0] + x_true[1][0]*dt + Q[0][0]*np.random.randn(1)[0]

            # updating velocity
            x_true[1][0] = x_true[1][0] + (-g - 0.5*rho*x_true[1][0]*np.abs(x_true[1][0])*(Cd*A)/x_true[2][0])*dt + Q[1][1]*np.random.rand(1)[0]

            # mass is just the dry mass
            x_true[2][0] = m_dry + Q[2][2]*np.random.randn(1)[0]

        # generate noisy measurement
        z = np.array([[x_true[0][0]], [x_true[1][0]]]) + np.sqrt(R) @ np.random.randn(2,1)

        # EKF predicition step
        x_pred = x_est + np.array([[x_est[1][0]*dt],
                                   [(-g - 0.5*rho*x_est[1][0]*np.abs(x_est[1][0])*(Cd*A)/x_est[2][0]+ x_est[1][0]/np.abs(x_est[1][0])*(mfr*u/x_est[2][0]))*dt],
                                   [(-mfr)*dt]])

        # state transition matrix
        F = np.array([[0, dt, 0],
                      [0, 
                       (-rho*((x_est[1][0])**2/(np.abs(x_est[1][0])))*((Cd*A)/(x_est[2][0])))*dt, 
                       (-0.5*rho*x_est[1][0]*np.abs(x_est[1][0])*(-Cd*A/(x_est[2][0]**2)) + (x_est[1][0]/np.abs(x_est[1][0]))*(-mfr*u/(x_est[2][0]**2)))*dt
                        ],
                      [0, 0, 0]])
        
        P_pred = F @ P @ F.T + Q

        # Observation matrix
        H = np.array([[1, 0, 0], [0, 1, 0]])

        # Compute Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)

        x_est = x_pred + K @ (z - np.array([[x_pred[0][0]], [x_pred[1][0]]]))

        P = (np.eye(3) - K @ H) @ P_pred

        # store values
        x_store[0][i] = x_true[0][0]
        x_store[1][i] = x_true[1][0]
        x_store[2][i] = x_true[2][0]
        x_est_store[0][i] = x_est[0][0]
        x_est_store[1][i] = x_est[1][0]
        x_est_store[2][i] = x_est[2][0]


    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1.plot(t, x_store[0,:], color='r') 
    ax1.plot(t, x_est_store[0,:], color='b', linestyle='--')
    ax1.set_xlabel('Time(s)')
    ax1.set_ylabel('Altitude (m)')
    ax2.plot(t, x_store[1,:], color='r') 
    ax2.plot(t, x_est_store[1,:], color='b', linestyle='--')
    ax2.set_xlabel('Time(s)')
    ax2.set_ylabel('Velocity (m)')
    ax3.plot(t, x_store[2,:], color='r')
    ax3.plot(t, x_est_store[2,:], color='b', linestyle='--')
    ax3.set_xlabel('Time(s)')
    ax3.set_ylabel('Mass (m)')
    plt.show()

def main():
    discrete_ekf()

if __name__ == "__main__":
    main()
