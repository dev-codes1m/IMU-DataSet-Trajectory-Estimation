import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to load IMU dataset
def load_imu_data(file_path):
    return pd.read_csv(file_path, chunksize=10000)  # Load in chunks to reduce memory usage

# Function to derive trajectory from IMU data
def derive_trajectory(imu_data):
    # Convert quaternion to Euler angles
    euler_angles = imu_data.apply(lambda row: quaternion_to_euler(row[['Quat_0', 'Quat_1', 'Quat_2', 'Quat_3']]), axis=1)
    euler_angles = np.array(euler_angles.tolist())

    # Integrate Euler angles to get orientation
    orientation = np.zeros((len(euler_angles), 3))
    for i in range(1, len(euler_angles)):
        orientation[i] = orientation[i-1] + euler_angles[i] * 0.01  # 100Hz -> 0.01s

    # Integrate acceleration to get position
    position = np.zeros((len(imu_data), 3))
    for i in range(1, len(imu_data)):
        acc = imu_data[['Acc_x', 'Acc_y', 'Acc_z']].iloc[i]
        position[i] = position[i-1] + acc * 0.01  # 100Hz -> 0.01s

    return position

# Function to convert quaternion to Euler angles
def quaternion_to_euler(quaternion):
    w, x, y, z = quaternion
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.array([roll, pitch, yaw])

# Streamlit app
st.title("IMU Trajectory Derivation")

# Load IMU dataset
file_path = st.file_uploader("Upload IMU dataset (CSV)", type="csv")
if file_path is not None:
    imu_data = pd.concat([chunk for chunk in load_imu_data(file_path)], ignore_index=True)  # Concatenate chunks

    # Derive trajectory
    trajectory = derive_trajectory(imu_data)

    # Plot trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    st.pyplot(fig)

    # Display Euler angles and quaternion values
    st.write("Euler Angles:")
    st.write(imu_data[['Euler_x', 'Euler_y', 'Euler_z']])
    st.write("Quaternion Values:")
    st.write(imu_data[['Quat_0', 'Quat_1', 'Quat_2', 'Quat_3']])