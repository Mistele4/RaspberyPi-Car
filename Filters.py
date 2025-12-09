import numpy as np
from ahrs.filters import Madgwick
from quaternion import quaternion, as_quat_array
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

'''
This file contains the Kalman and Madgwick filter implementations for the RPi car
The Madgwick filter converts relative accel and gyro data into quaternary orientation data
The Kalman filter uses the orientation data to help estimate position over time
Bot filters are run retroactively in this case to reduce processing load on the RPi
Both filters also work to clean the date to create more accurate results
because of thi, they tent to work better retroactively rather than live due to
the larger data sets available at the time of filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
I was lucky to find some very simpl pre-existing libraries for both of these
Relatively complex filters.
The logic and design for the raspberry pi car controll and data collection is all
my own, but the filter implementations are taken from the libraries to prevent the need
for some exceedingly complex calculus that I cannot imagine is much fun to code.
'''


def madgwick_filter(imu_data):
    '''Returns quaternions from Madgwick filter given IMU data to create a global frame of reference'''

    # Applys the Madgick filter to the IMU data
    madgwick = Madgwick(gyr = imu_data.get_gyro(), acc = imu_data.get_accel(), freq=562.5)
    Q = madgwick.Q
    return Q


def rotate_accel(accel, quat):
    '''Takes raw accel data and quaternions to return global frame accel data'''

    # quat in (n, 4) format [w,x,y,z]
    # accel in (n, 3) format [ax,ay,az]

    rotations = R.from_quat(quat[:, [1,2,3,0]])  # scipy expects [x,y,z,w]
    return rotations.apply(accel)


def smooth_accel(accel, window=5):
    '''Creates a simple moving average to smooth accel data and decrease the effect of outliers'''
    
    # creates a {window}-sized array of equal weights
    kernel = np.ones(window) / window
    # creates an empty array to hold smoothed data
    smooth = np.empty_like(accel)
    for i in range(3):
        # convolves each axis with the kernal to create a moving average and smooth data
        smooth[:, i] = np.convolve(accel[:, i], kernel, mode='same')
    return smooth


class Simple_Accel_Kalman:

    def __init__(self, imu):
        '''initiallizes a simple Kalman filter for 3D position and velocity estimation from accel data'''

        self.imu_data = imu
        self.kalman = KalmanFilter(dim_x=9, dim_z=3) # states: [rx, ry, rz, vx, vy, vz, ax, ay, az]

        # dt: defines time step
        self.dts = np.diff(imu.get_timestamp(), prepend=imu.get_timestamp()[0])

        self._set_initial_values()

        self.idx = 0 # to track time step index

    def _build_matrices(self, dt):
        '''Builds the state transition matrix for the Kalman filter'''
        
        # F: state transition matrix [pos, vel, accel]
        F = np.eye(9) 
        F[0:3, 3:6] = np.diag([dt,dt,dt])                   # Position is updated from vel
        F[0:3, 6:9] = 0.5 * np.diag([dt**2,dt**2,dt**2])    # Position is updated from accel
        F[3:6, 6:9] = np.diag([dt,dt,dt])                   # Velocity is updated from accel

        self.kalman.F = F


    def _set_initial_values(self):
        '''Sets initial values for the Kalman filter matrices for states, uncertainty, and noises'''

        self.kalman.x = np.zeros((9, 1))             # [px,py,pz,  vx,vy,vz,  ax,ay,az]
        self.kalman.P *= 1                           # initial uncertainty
        self.covariance = self.imu_data.stdevs ** 2  # variance from accel stdevs
        self.kalman.R = np.diag(self.covariance)     # measurement noise (IMU accel)
        self.kalman.Q = 0.1 * np.eye(9)              # process noise
        # H: Accel is directly observed
        self.kalman.H = np.zeros((3, 9))             # observation matrix
        self.kalman.H[:, 6:9] = np.eye(3)


    def predict_update(self, global_accel):
        '''Creates prediction and update steps for Kalman filter given global accel data'''

        # Rebuilds F matric for each dt
        dt = self.dts[self.idx]
        self._build_matrices(dt)

        z = global_accel.reshape(3,1)

        self.kalman.predict()
        self.kalman.update(z)

        self.idx += 1 # increment time step index
        return self.kalman.x[0:3, 0]  # return position only


def run_filters(imu_data):
    '''Runs the Madgwick and Kalman filters on IMU data to return final position and position over time'''
    
    # Runs madgwick filter to get quaternions
    quats = madgwick_filter(imu_data)

    print(f"Kalman filter time step: {1/562.5} seconds")
    print(f"Kalman filter frequency: {562.5} Hz")
    print(f"Measurement timespan: {imu_data.get_timespan()}")
    print(f"Number of samples: {imu_data.get_length()}")
    print(f"Estimated Hz: {imu_data.get_length() / imu_data.get_timespan()} vs Expected Hz: 562.5")


    # print("Quats shape:", quats.shape)
    # print("Any zero rows?:", np.any(np.linalg.norm(quats, axis=1) == 0))
    # print("First 5 quats:\n", quats[:5])

    # Rotates accel data into global frame
    global_accel = rotate_accel(imu_data.get_accel(), quats)
    print(f"Global Accel shape: {global_accel.shape}")
    print(f"First 5 Global Accel:\n {global_accel[:5]}")
    print(f"Global Accel mean BEFORE gravity is removed: {np.mean(global_accel, axis=0)}")
    
    # Remove gravity from globalized accel data
    gravity = np.array([0, 0, 9.80665])  # m/s^2
    global_accel -= gravity  # m/s^2
    print(f"Global Accel mean AFTER gravity is removed: {np.mean(global_accel, axis=0)}")

    # Smooth the global accel data
    global_accel = smooth_accel(global_accel, window=5)

    positions = []
    kalman = Simple_Accel_Kalman(imu_data)
    for accel in global_accel:
        pos  = kalman.predict_update(accel)
        positions.append(pos)

    final_position = positions[-1]

    vel_wo_k = np.cumsum(global_accel*(1/562.5), axis=0)
    pos_wo_k = np.cumsum(vel_wo_k*(1/562.5), axis=0)
    print(f"Final position estimate no kalman: {pos_wo_k[-1]}")
    
    return final_position, np.array(positions)


## Make sure to transfer to raspberry pi and test with real data! ##
    


    