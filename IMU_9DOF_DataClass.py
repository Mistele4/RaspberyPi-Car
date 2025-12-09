import time
import sys
import qwiic_icm20948
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise



class IMUDataReadings:

    def __init__(self):
        self.accel = np.empty((0, 3))
        self.gyro = np.empty((0, 3))
        self.mag = np.empty((0, 3))
        self.timestamp = np.array([])
        self.x = 0
        self.y = 1
        self.z = 2
        
        # Conversion facters from raw signed int data to common units
        # Accel: to mGs to Gs to m/s^2
        self.accel_conv = 9.80665 / (16.384 * 1000)
        # Gyro: to deg/sec to rad/sec
        self.gyro_conv = (1 / 131) * (np.pi / 180)
        #Mag: to microTesla
        self.mag_conv = .15 # / 1000 for miliTesla for Madgwick filter
    
    def __getlength__(self):
        return len(self.accel)
    
    def add_reading(self, accel, gyro, mag):
        self.accel = np.append(self.accel, [accel], axis=0)
        self.gyro = np.append(self.gyro, [gyro], axis=0)
        self.mag = np.append(self.mag, [mag], axis=0)
        self.timestamp = np.append(self.timestamp, time.time())

    def get_timespan(self):
        if len(self.timestamp) < 2:
            return None
        return self.timestamp[-1] - self.timestamp[0]

    def get_accel(self, col=-1):
        if col == -1:
            return self.accel
        return self.accel[:,col]
    
    def get_gyro(self, col=-1):
        if col == -1:
            return self.gyro
        return self.gyro[:,col]
    
    def get_mag(self, col=-1):
        if col == -1:
            return self.mag
        return self.mag[:,col]

    # This method aims to just view the raw data that 9DOF IMU reads statically as a test and/or to be used for calibration
    def static_sensor_test(self, duration=10, hz=562.5):
        # 3 data retrieval frequencies for Gyro [9000, 1125, 562.5] Hz
        # 3 for Accel [4500, 1125, 562.5] Hz
        # 1 for Mag [100] 

        # instantiates sparkfun 9DOF IMU class
        IMU = qwiic_icm20948.QwiicIcm20948()

        # tests IMU is connected to the RPi (IMU is integrated, so there are serious problems if not)
        if IMU.connected == False:
            print("The Qwiic ICM20948 device isn't connected to the system. Please check your connection", \
                  file=sys.stderr)
            return
    
        # allows IMU to be called and functions to be used
        IMU.begin()
    
        #data collection, conversion and storage
        start_time = time.time()
        while (time.time() - start_time) < duration:
            # from the docs ive read, `.dataReady()` should return bool True if any of the sensors (A,M,G) have readable data
            # and the using `.getAgmt()` should repeat values if the other sensors do not
            if IMU.dataReady():
                IMU.getAgmt()
                A = np.array([IMU.axRaw, IMU.ayRaw, IMU.azRaw]) * self.accel_conv
                G = np.array([IMU.gxRaw, IMU.gyRaw, IMU.gzRaw]) * self.gyro_conv
                M = np.array([IMU.mxRaw, IMU.myRaw, IMU.mzRaw]) * self. mag_conv
                self.add_reading(A, G, M)
                time.sleep(1/hz)
        

        # in hindsight, thers probably a much better way to do this, 
        # however, it works for my short term test purposes here,
        # but will hopefully be more concise in the final project
        ax, ay, az = self.get_accel(self.x), self.get_accel(self.y), self.get_accel(self.z)
        gx, gy, gz = self.get_gyro(self.x), self.get_gyro(self.y), self.get_gyro(self.z)
        mx, my, mz = self.get_mag(self.x), self.get_mag(self.y), self.get_mag(self.z)

        # Collect means
        ax_mean, ay_mean, az_mean = np.mean(ax), np.mean(ay), np.mean(az)
        gx_mean, gy_mean, gz_mean = np.mean(gx), np.mean(gy), np.mean(gz)
        mx_mean, my_mean, mz_mean = np.mean(mx), np.mean(my), np.mean(mz)
        mean_data = np.array([[ax_mean, ay_mean, az_mean], [gx_mean, gy_mean, gz_mean], [mx_mean, my_mean, mz_mean]])
        # print('mean data: ', mean_data)

        # Collect STDs
        ax_std, ay_std, az_std = np.std(ax), np.std(ay), np.std(az)
        gx_std, gy_std, gz_std = np.std(gx), np.std(gy), np.std(gz)
        mx_std, my_std, mz_std = np.std(mx), np.std(my), np.std(mz)

        # Collect tolerances
        ax_tol = max(abs(max(ax)-ax_mean), abs(min(ax)-ax_mean))
        ay_tol = max(abs(max(ay)-ay_mean), abs(min(ay)-ay_mean))
        az_tol = max(abs(max(az)-az_mean), abs(min(az)-az_mean))
        gx_tol = max(abs(max(gx)-gx_mean), abs(min(gx)-gx_mean))
        gy_tol = max(abs(max(gy)-gy_mean), abs(min(gy)-gy_mean))
        gz_tol = max(abs(max(gz)-gz_mean), abs(min(gz)-gz_mean))
        mx_tol = max(abs(max(mx)-mx_mean), abs(min(mx)-mx_mean))
        my_tol = max(abs(max(my)-my_mean), abs(min(my)-my_mean))
        mz_tol = max(abs(max(mz)-mz_mean), abs(min(mz)-mz_mean))

        # Formatted output
        out = f'''
            \t  X   \t\tY   \t\tZ
Accel. avg |\t{round(ax_mean, 1)}\t+/-{round(ax_tol, 1)}\t{round(ay_mean, 1)}\t+/-{round(ay_tol, 1)}\t{round(az_mean, 1)}\t+/-{round(az_tol, 1)}
       std |\t{round(ax_std, 1)}\t\t{round(ay_std, 1)}\t\t{round(az_std, 1)}
---------------------------------------------------------------------------------------------------------
Gyro.  avg |\t{round(gx_mean, 1)}\t+/-{round(gx_tol, 1)}\t{round(gy_mean, 1)}\t+/-{round(gy_tol, 1)}\t{round(gz_mean, 1)}\t+/-{round(gz_tol, 1)}
       std |\t{round(gx_std, 1)}\t\t{round(gy_std, 1)}\t\t{round(gz_std, 1)}
---------------------------------------------------------------------------------------------------------
Mag.   avg |\t{round(mx_mean, 1)}\t+/-{round(mx_tol, 1)}\t{round(my_mean, 1)}\t+/-{round(my_tol, 1)}\t{round(mz_mean, 1)}\t+/-{round(mz_tol, 1)}
       std |\t{round(mx_std, 1)}\t\t{round(my_std, 1)}\t\t{round(mz_std, 1)}
    '''
        print(out)

        # returns mean data to be used for calibration
        return mean_data
    
    # This function aims to take data from the static test and use it to adjust for sensor bias
    def sensor_cal_test(self, means, duration=15, hz=562.5):

        IMU = qwiic_icm20948.QwiicIcm20948()
        if IMU.connected == False:
            print("The Qwiic ICM20948 device isn't connected to the system. Please check your connection", \
                  file=sys.stderr)
            return
    
        self.accel = np.empty((0, 3))
        self.gyro = np.empty((0, 3))
        self.mag = np.empty((0, 3))
        self.timestamp = np.array([])
        
        IMU.begin()
        start_time = time.time()
        while (time.time() - start_time) < duration:
            if IMU.dataReady():
                IMU.getAgmt()
                A = [(IMU.axRaw * self.accel_conv - means[0,0]), (IMU.ayRaw * self.accel_conv - means[0,1]), (IMU.azRaw * self.accel_conv - means[0,2])]
                G = [(IMU.gxRaw * self.gyro_conv - means[1,0]), (IMU.gyRaw * self.gyro_conv - means[1,1]), (IMU.gzRaw * self.gyro_conv- means[1,2])]
                M = [(IMU.mxRaw * self.mag_conv), (IMU.myRaw * self.mag_conv), (IMU.mzRaw * self.mag_conv)]
                self.add_reading(A, G, M)
                time.sleep(1/hz)
        
        outdata = np.array([self.get_accel(), self.get_gyro(), self.get_mag()])

        # pulls data for analysis
        ax, ay, az = self.get_accel(self.x), self.get_accel(self.y), self.get_accel(self.z)
        gx, gy, gz = self.get_gyro(self.x), self.get_gyro(self.y), self.get_gyro(self.z)
        mx, my, mz = self.get_mag(self.x), self.get_mag(self.y), self.get_mag(self.z)


        # Collect means
        ax_mean, ay_mean, az_mean = np.mean(ax), np.mean(ay), np.mean(az)
        gx_mean, gy_mean, gz_mean = np.mean(gx), np.mean(gy), np.mean(gz)
        mx_mean, my_mean, mz_mean = np.mean(mx), np.mean(my), np.mean(mz)
       
        # Collect tolerances
        ax_tol = max(abs(max(ax)-ax_mean), abs(min(ax)-ax_mean))
        ay_tol = max(abs(max(ay)-ay_mean), abs(min(ay)-ay_mean))
        az_tol = max(abs(max(az)-az_mean), abs(min(az)-az_mean))
        gx_tol = max(abs(max(gx)-gx_mean), abs(min(gx)-gx_mean))
        gy_tol = max(abs(max(gy)-gy_mean), abs(min(gy)-gy_mean))
        gz_tol = max(abs(max(gz)-gz_mean), abs(min(gz)-gz_mean))
        mx_tol = max(abs(max(mx)-mx_mean), abs(min(mx)-mx_mean))
        my_tol = max(abs(max(my)-my_mean), abs(min(my)-my_mean))
        mz_tol = max(abs(max(mz)-mz_mean), abs(min(mz)-mz_mean))

        # Formatted output
        out = f'''
            \t  X   \t\t\tY   \t\t\tZ
Accel. avg |\t{round(ax_mean, 1)}\t+/-{round(ax_tol, 1)}\t{round(ay_mean, 1)}\t+/-{round(ay_tol, 1)}\t{round(az_mean, 1)}\t+/-{round(az_tol, 1)}
Gyro.  avg |\t{round(gx_mean, 1)}\t+/-{round(gx_tol, 1)}\t{round(gy_mean, 1)}\t+/-{round(gy_tol, 1)}\t{round(gz_mean, 1)}\t+/-{round(gz_tol, 1)}
Mag.   avg |\t{round(mx_mean, 1)}\t+/-{round(mx_tol, 1)}\t{round(my_mean, 1)}\t+/-{round(my_tol, 1)}\t{round(mz_mean, 1)}\t+/-{round(mz_tol, 1)}
    '''
        print(out)
        
        # returns output data string
        return outdata

