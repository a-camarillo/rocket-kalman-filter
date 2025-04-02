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

def main():
    t, dt = np.linspace(0,1000, num=10000, retstep=True)

    x = t*0
    v = t*0
    m = t*0

    mfr = 2555

    x[0] = 20
    v[0] = 2
    m[0] = m0

    for i in range(len(t)-1):
        x[i+1] = x[i] + v[i]*dt
        v[i+1] = v[i] + (-g - 0.5*rho*v[i]*np.abs(v[i])*(Cd*A)/m[i]+ v[i]/np.abs(v[i])*(mfr*u/m[i]))*dt
        if m[i] < m_dry:
            # set dry mass conditions
            m[i+1] = m_dry
            # mass flow rate goes to zero since there's no fuel
            mfr = 0
        else:
            m[i+1] = m[i] + (-mfr)*dt
        
        if x[i+1] == 0:
            break
        

    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1.plot(t,x)
    ax2.plot(t,v)
    ax3.plot(t,m)
    ax1.set_ylabel('Position')
    ax2.set_ylabel('Velocity')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Mass')

    plt.show()

if __name__ == "__main__":
    main()
