from AGDataClass import IMUDataReadings
from Filters import *
import numpy as np
import time
import sys

imu = IMUDataReadings()

cal = input("Would you like to static Calibrate? Y/N: ").lower()
if cal == "n":
    sys.exit(0)
print("Calibrating in:")
print("5...")
time.sleep(1)
print("4...")
time.sleep(1)
print("3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)
print("Calibrating...")
imu.static_sensor_data()
print("Caliration Complete")

run = input("Are you ready to take readings? Y/N: ").lower()
if run == "n":
    sys.exit(0)
print("Starting in:")
print("5...")
time.sleep(1)
print("4...")
time.sleep(1)
print("3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)
print("Taking reading. press ctrl+c to conclude")
imu.take_reading()
print("Data collection complete")
print(f"Accel shape: {imu.get_accel().shape}")
print(f"Gyro Shape: {imu.get_gyro().shape}")
print(f"Accel sample (First 5 rows):\n{imu.get_accel()[:5]}")
print(f"Accel mean (all rows): {np.mean(imu.get_accel(), axis=0)}")

pos, pos_array = run_filters(imu)
print(f"Final position with Kalman filter: {pos}")