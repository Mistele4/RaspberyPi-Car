import numpy as np
from ahrs.filters import Madgwick
from quaternion import quaternion, as_quat_array
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

###############
#### Notes ####
###############
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This file contains the Kalman and Madgwick filter implementations for the RPi car
# The Madgwick filter converts relative accel and gyro data into quaternary orientation data
# The Kalman filter uses the orientation data to help estimate position over time
# Bot filters are run retroactively in this case to reduce processing load on the RPi
# Both filters also work to clean the date to create more accurate results
# because of thi, they tent to work better retroactively rather than live due to
# the larger data sets available at the time of filtering
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# I was lucky to find some very simpl pre-existing libraries for both of these
# Relatively complex filters.
# The logic and design for the raspberry pi car controll and data collection is all
# my own, but the filter implementations are taken from the libraries to prevent the need
# for some exceedingly complex calculus that I cannot imagine is fun to code.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def madgwick_filter(imu_data):
    '''Returns quaternions from Madgwick filter given IMU data to create a global frame of reference'''

    # Applys the Madgick filter to the IMU data
    madgwick = Madgwick(gyr = imu_data.get_gyro(), acc = imu_data.get_accel(), freq=562.5)
    Q = madgwick.Q
    return Q


def rotate_accel(accel, quat):
    '''Takes raw accel data and quaternions to return global frame accel data'''

    # quat in (n, 4) format [qo, q1, q2, q3, q4]
    # accel in (n, 3) format [ax, ay, az]

    rotations = R.from_quat(quat[:, [1,2,3,0]])  # scipy expects [x,y,z,w]
    return rotations.apply(accel)


class Simple_Accel_Kalman:

    def __init__(self, hz=562.5):
        '''initiallizes a simple Kalman filter for 3D position and velocity estimation from accel data'''

        self.kalman = KalmanFilter(dim_x=9, dim_z=3) # states: [rx, ry, rz, vx, vy, vz, ax, ay, az]

        # dt: defines time step
        self.dt = 1/hz

        self._build_matrices()
        self._set_initial_values()

    def _build_matrices(self):
        '''Builds the state transition and observation matrices for the Kalman filter'''

        # F: state transition matrix [pos, vel, accel]
        F = np.eye(9)
        # Position is updated from vel
        F[0:3, 3:6] = np.diag([self.dt, self.dt, self.dt])
        # Position is updated from accel
        F[0:3, 6:9] = 0.5 * np.diag([self.dt**2, self.dt**2, self.dt**2])
        # Velocity is updated from accel
        F[3:6, 6:9] = np.diag([self.dt, self.dt, self.dt])

        self.kalman.F = F

        # H: Accel is directly observed
        self.kalman.H = np.zeros((3, 9))
        self.kalman.H[:, 6:9] = np.eye(3)


    def _set_initial_values(self):
        '''Sets initial values for the Kalman filter matrices for states, uncertainty, and noises'''

        self.kalman.x = np.zeros((9, 1))    # [px,py,pz,  vx,vy,vz,  ax,ay,az]
        self.kalman.P *= 1                  # initial uncertainty
        self.kalman.R = 0.1 * np.eye(3)     # measurement noise (IMU accel)
        self.kalman.Q = 0.01 * np.eye(9)    # process noise


    def predict_update(self, global_accel):
        '''Creates prediction and update steps for Kalman filter given global accel data'''

        z = global_accel.reshape(3,1)

        self.kalman.predict()
        self.kalman.update(z)

        return self.kalman.x[0:3, 0]  # return position only


def run_filters(imu_data):
    '''Runs the Madgwick and Kalman filters on IMU data to return final position and position over time'''
    
    # Runs madgwick filter to get quaternions
    quats = madgwick_filter(imu_data)

    # Rotates accel data into global frame
    global_accel = rotate_accel(imu_data.get_accel(), quats)
    
    # Remove gravity from globalized accel data
    gravity = np.array([0, 0, 9.80665])  # m/s^2
    global_accel -= gravity  # m/s^2
    

    positions = []
    kalman = Simple_Accel_Kalman(hz=562.5)
    for accel in global_accel:
        pos  = kalman.predict_update(accel)
        positions.append(pos)

    final_position = positions[-1]
    
    return final_position, np.array(positions)
    

    


    