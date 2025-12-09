import time
import sys
import qwiic_icm20948
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

###############
#### Notes ####
###############
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is a class very similar to the IMUData Class
# However this one omits magnetometer data reading and handling
# This is because the strength of earths magnetic field is very weak
# comparred to the magnetic interference caused by the motors
# (50-80uT compared to 100-1000s of uT)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The madgwick filters that will turn relative data into data with quaternary headings
# Is capable of working without magnetometer data, though it is less accurate causing slight yaw drift over time
# However, for the purposes of this project, the magnetometer data cannot be reliably parsed
# to create accurate readings of earths field, so the exclusion is necessary
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This class is designed to be used to retroactively create an estimate of relative position of the RPi compred to start
# It can be adjusted to allow live readings for pos and vel, but this increases complexity and strains the RPi
# Running the Madgwick and Kalman Filters after the fact on the entirety of the data also helps to create cleaner data
# and produce more accurate results
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class IMUDataReadings:

    def __init__(self):
        '''Initializes class variables for storing IMU data readings'''
        self.accel_list = []
        self.gyro_list = []
        self.timestamp_list = []
        self.accel = np.empty((0, 3))
        self.gyro = np.empty((0, 3))
        self.timestamp = np.array([])
        self.means = None
        self.x = 0
        self.y = 1
        self.z = 2
        
        # Conversion facters from raw signed int data to common units
        # Accel: to mGs to Gs to m/s^2
        self.accel_conv = 9.80665 / (16.384 * 1000)
        # Gyro: to deg/sec to rad/sec
        self.gyro_conv = (1 / 131) * (np.pi / 180)
        #Mag: to microTesla


    ##################
    #### Getters #####
    ##################

    def get_length(self):
        '''Returns number of readings stored'''
        return len(self.accel)


    def get_timespan(self):
        '''Returns total time span of readings in seconds'''
        if len(self.timestamp) < 2:
            return None
        return self.timestamp[-1] - self.timestamp[0]


    def get_accel(self, col=-1):
        '''Returns measured acceleration data'''
        if col == -1:
            return self.accel
        return self.accel[:,col]
    

    def get_gyro(self, col=-1):
        '''Returns measured gyroscopic data'''
        if col == -1:
            return self.gyro
        return self.gyro[:,col]
    

    def add_reading(self, accel, gyro):
        '''Adds a new reading to instance stored data'''
        self.accel_list.append(accel)
        self.gyro_list.append(gyro)
        self.timestamp_list.append(time.time())


    def static_sensor_data(self, duration=10, hz=562.5):
        '''Runs a static test to collect sensor bias data for calibration. 
        Assumes IMU is stationary and z-axis aligned with gravity.'''

        # clears instance data storage for calibtration
        self.accel = np.empty((0, 3))
        self.gyro = np.empty((0, 3))
        self.timestamp = np.array([])
        self.period = 1 / hz

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
                start = time.perf_counter()
                IMU.getAgmt()
                A = [IMU.axRaw, IMU.ayRaw, IMU.azRaw]
                G = [IMU.gxRaw, IMU.gyRaw, IMU.gzRaw]
                self.add_reading(A, G)
                elapsed = time.perf_counter() - start
                time.sleep(max(0, self.period - elapsed))
        

        # in hindsight, thers probably a much better way to do this, 
        # however, it works for my short term test purposes here,
        # but will hopefully be more concise in the final project
        ax, ay, az = self.get_accel(self.x), self.get_accel(self.y), self.get_accel(self.z)
        gx, gy, gz = self.get_gyro(self.x), self.get_gyro(self.y), self.get_gyro(self.z)

        # Collect means
        ax_mean, ay_mean, az_mean = np.mean(ax), np.mean(ay), np.mean(az)
        gx_mean, gy_mean, gz_mean = np.mean(gx), np.mean(gy), np.mean(gz)
        # Saves means for calibration of Accel and Gyro
        mean_data = np.array([[ax_mean, ay_mean, az_mean], [gx_mean, gy_mean, gz_mean]])
        self.means = mean_data
    

    def apply_cal(self, means):
        '''Removes sensor bias from stored data using means from static test'''
        # remove gravity from z axis to only adjust sensor bias
        # assumes static calibration with z axis aligned with gravity
        # but allows for data reading in any orientation
        means[0,2] -= 9.80665
        self.accel = self.accel - means[0]
        self.gyro = self.gyro - means[1]


    def take_reading(self, hz=562.5):
        '''Runs sensors to collect data untill keyboard interrupt'''
        # clears instance data storage for calibtration
        self.accel = np.empty((0, 3))
        self.gyro = np.empty((0, 3))
        self.timestamp = np.array([])
        self.accel_list = []
        self.gyro_list = []
        self.timestamp_list = []
        self.period = 1/hz

        # instantiates sparkfun 9DOF IMU class
        IMU = qwiic_icm20948.QwiicIcm20948()

        # tests IMU is connected to the RPi (IMU is integrated, so there are serious problems if not)
        if IMU.connected == False:
            print("The Qwiic ICM20948 device isn't connected to the system. Please check your connection", \
                  file=sys.stderr)
            return
    
        # allows IMU to be called and functions to be used
        IMU.begin()
    

        # Read data until keyboard interrupt
        # Add in Motor control!!!!!!!!!!!!!!!!!!
        try:
            while True:
                # from the docs ive read, `.dataReady()` should return bool True if any of the sensors (A,M,G) have readable data
                # and the using `.getAgmt()` should repeat values if the other sensors do not
                if IMU.dataReady():
                    start = time.perf_counter()
                    IMU.getAgmt()
                    A = [IMU.axRaw, IMU.ayRaw, IMU.azRaw]
                    G = [IMU.gxRaw, IMU.gyRaw, IMU.gzRaw]
                    self.add_reading(A, G)
                    elapsed = time.perf_counter() - start
                    time.sleep(max(0, self.period - elapsed)) # consider changing to subtract runtime by checking time at begining and end of ea loop

        except KeyboardInterrupt:
            print("\nEnding Data Read")
            self.accel = np.array(self.accel_list) * self.accel_conv
            self.gyro = np.array(self.gyro_list) * self.gyro_conv
            self.timestamp = np.array(self.timestamp_list)
            self.apply_cal(self.means)
            pass

    
    def sensor_cal_test(self, duration=15, hz=562.5):
        '''Test to make sure that calibration properly zeroes data'''

        means = self.means
        IMU = qwiic_icm20948.QwiicIcm20948()
        if IMU.connected == False:
            print("The Qwiic ICM20948 device isn't connected to the system. Please check your connection", \
                  file=sys.stderr)
            return

        # clears instance data storage for calibtration
        self.accel = np.empty((0, 3))
        self.gyro = np.empty((0, 3))
        self.timestamp = np.array([])
        
        IMU.begin()
        start_time = time.time()
        while (time.time() - start_time) < duration:
            if IMU.dataReady():
                IMU.getAgmt()
                A = [(IMU.axRaw * self.accel_conv - means[0,0]), (IMU.ayRaw * self.accel_conv - means[0,1]), (IMU.azRaw * self.accel_conv - means[0,2])]
                G = [(IMU.gxRaw * self.gyro_conv - means[1,0]), (IMU.gyRaw * self.gyro_conv - means[1,1]), (IMU.gzRaw * self.gyro_conv- means[1,2])]
                self.add_reading(A, G)
                time.sleep(1/hz)
        
        outdata = np.array([self.get_accel(), self.get_gyro()])

        # pulls data for analysis
        ax, ay, az = self.get_accel(self.x), self.get_accel(self.y), self.get_accel(self.z)
        gx, gy, gz = self.get_gyro(self.x), self.get_gyro(self.y), self.get_gyro(self.z)


        # Collect means
        ax_mean, ay_mean, az_mean = np.mean(ax), np.mean(ay), np.mean(az)
        gx_mean, gy_mean, gz_mean = np.mean(gx), np.mean(gy), np.mean(gz)
       
        # Collect tolerances
        ax_tol = max(abs(max(ax)-ax_mean), abs(min(ax)-ax_mean))
        ay_tol = max(abs(max(ay)-ay_mean), abs(min(ay)-ay_mean))
        az_tol = max(abs(max(az)-az_mean), abs(min(az)-az_mean))
        gx_tol = max(abs(max(gx)-gx_mean), abs(min(gx)-gx_mean))
        gy_tol = max(abs(max(gy)-gy_mean), abs(min(gy)-gy_mean))
        gz_tol = max(abs(max(gz)-gz_mean), abs(min(gz)-gz_mean))

        # Formatted output
        out = f'''
(X,Y,Z)
Accel. avg |\t{round(ax_mean, 1)}\t+/-{round(ax_tol, 1)}\t{round(ay_mean, 1)}\t+/-{round(ay_tol, 1)}\t{round(az_mean, 1)}\t+/-{round(az_tol, 1)}
Gyro.  avg |\t{round(gx_mean, 1)}\t+/-{round(gx_tol, 1)}\t{round(gy_mean, 1)}\t+/-{round(gy_tol, 1)}\t{round(gz_mean, 1)}\t+/-{round(gz_tol, 1)}
    '''
        print(out)
        
        # returns output data string
        return outdata

  