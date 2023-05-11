import pandas as pd
import numpy as np
import json
from scipy import signal
from scipy.signal import butter, filtfilt
from matplotlib import pyplot as plt
#import preproc


def preproc_demog(Demog_raw):
    # Drop variables I won't focus on
    Demog = Demog_raw.copy()
    Demog.drop(Demog.iloc[:, [*[0, 1, 5, 6, 8], *list(range(11, 13)), *list(range(15, 21)), *list(range(22, 26)),
                              *list(range(27, 30)), *list(range(31, 33))]],
               inplace=True, axis=1)

    # Exploring the descriptive statistics of the variables
    Demog[["age", "gender", "health-history", "professional-diagnosis"]].describe(include="all")
    # Let's map "Male", "Female" to {0,1} and "False", "True" to {0,1}
    Demog['gender'] = Demog['gender'].map({'Male': 0, 'Female': 1})
    Demog['professional-diagnosis'] = Demog['professional-diagnosis'].map({False: 0, True: 1})

    # Dealing with missing values (age, gender, professional_diagnosis)
    Demog[["age", "gender", "professional-diagnosis"]].isnull().sum()
    # No. missing values: age = 52, gender = 23, professional_diagnosis = 137
    Demog = Demog.dropna(subset=['age', 'gender', 'professional-diagnosis'])
    Demog[["age", "gender", "health-history", "professional-diagnosis"]].describe(include="all")
    # !! check variables with invalid values!
    Demog = Demog[Demog['gender'].isin([0, 1])]
    Demog = Demog[Demog['professional-diagnosis'].isin([0, 1])]

    # Exploring the PDFs
    # sns.displot(Demog, x="age", hue="professional-diagnosis", legend=False)
    # plt.legend(labels=["pd", "hc"])
    # plt.show()
    # sns.displot(Demog, x="gender", hue="professional-diagnosis", legend=False)
    # plt.show()

    return Demog


def demog_gait(Demog, Walking_raw):
    # Cleaning of the walking data with demographics
    Walking_raw = pd.read_csv('Walking.csv')

    # Merge demographic and walking data based on healthCode
    Walking = pd.merge(Demog, Walking_raw, on=['healthCode'])
    Walking.drop(Walking.iloc[:, [*[0, 2, 11, 12, 15, 16]]], inplace=True, axis=1)

    # Delete inconsistencies
    ind_del = np.array(np.where((Walking['professional-diagnosis'] == 0) & (
        Walking['medTimepoint'].str.contains('Just after Parkinson medication'))))
    ind_del = np.append(ind_del, np.array(np.where((Walking['professional-diagnosis'] == 0) & (
        Walking['medTimepoint'].str.contains('Immediately before Parkinson medication')))))
    ind_del = np.append(ind_del, np.where(
        (Walking['professional-diagnosis'] == 0) & (pd.Series(~np.isnan(Walking['diagnosis-year'].tolist())))))
    ind_del = np.append(ind_del, np.where((Walking['professional-diagnosis'].isin([0])) & Walking['surgery']))
    ind_del = np.append(ind_del,
                        np.where((Walking['professional-diagnosis'].isin([0])) & Walking['deep-brain-stimulation']))

    a = pd.Series(~np.isnan(Walking['medication-start-year'].tolist()))
    b = ~Walking['medication-start-year'].isin([0])
    c = Walking['professional-diagnosis'].isin([0])
    ind_del = np.append(ind_del, np.where(a & b & c))
    del a, b, c

    ind_del = np.unique(ind_del)  # since many are repeated
    Walking.drop(ind_del, inplace=True)
    del ind_del

    # Exploring the PDFs
    # sns.displot(Walking, x="age", hue="professional-diagnosis", legend=False)
    # plt.legend(labels=["pd", "hc"])
    # plt.show()
    # sns.displot(Walking, x="gender", hue="professional-diagnosis", legend=False)
    # plt.legend(labels=["pd", "hc"])
    # plt.show()
    # it is clearly so biased towards young and male

    return Walking


def LPfilter(data, fs):
    # Butterworth low pass filter 4th order cutoff freq 20 Hz

    # filter parameters
    nyquist = fs / 2
    fc = 20
    
    # create filter coefficients
    b, a = signal.butter(4, fc / nyquist, btype='low', analog=False)

    # Apply the filter
    data_filt = filtfilt(b, a, data)

    return data_filt


def HPfilter(data, fs):
    # Butterworth high pass filter 3th order cutoff freq 0.3 Hz

    nyquist = fs / 2
    fc = 0.3

    # create filter coefficients
    b, a = signal.butter(3, fc / nyquist, btype='high', analog=False)

    # Apply the filter
    data_filt = filtfilt(b, a, data)

    return data_filt


def linearacceleration(data):

    data = pd.DataFrame(data)
    data.head()

    # sample frequency
    t = data['timestamp'].to_numpy()
    t_inter = t[1:] - t[0:-1]
    fs = 1 / np.mean(t_inter)
    fs = np.around(fs, -2)  # not to get decimals

    # Attitude of the smartphone
    attitude = data['attitude']
    x, y, z, w = np.array([]), np.array([]), np.array([]), np.array([])
    for i in attitude:
        x = np.append(x, i['x'])
        y = np.append(y, i['y'])
        z = np.append(z, i['z'])
        w = np.append(w, i['w'])

    # Quaternion representing the attitude of the smartphone
    q = np.array([x, y, z, w])

    q_filt = LPfilter(q, fs) # LPF

    # # Plot attitude
    # plt.figure()
    # plt.plot(q_filt[0], label='x');
    # plt.plot(q_filt[1], label='y');
    # plt.plot(q_filt[2], label='z');
    # plt.plot(q_filt[3], label='w')
    # plt.xlim(0, len(q_filt[0]) - 1)
    # plt.title('Attitude data (x, y, z, w)')
    # plt.xlabel('Samples')
    # plt.legend()

    # User acceleration in Gs units
    UA = data['userAcceleration']
    x_accel, y_accel, z_accel = np.array([]), np.array([]), np.array([])
    for i in UA:
        x_accel = np.append(x_accel, i['x'])
        y_accel = np.append(y_accel, i['y'])
        z_accel = np.append(z_accel, i['z'])

    user_acceleration = np.array([x_accel, y_accel, z_accel])

    user_acceleration_filt = LPfilter(user_acceleration, fs) # LPF

    # # Plot user acceleration
    # plt.figure()
    # plt.plot(user_acceleration_filt[0], label='x');
    # plt.plot(user_acceleration_filt[1], label='y');
    # plt.plot(user_acceleration_filt[2], label='z')
    # plt.xlim(0, len(user_acceleration_filt[0]) - 1)
    # plt.title('User acceleration')
    # plt.xlabel('Samples')
    # plt.legend()

    # Gravity in Gs units
    Grav = data['gravity']
    x_gravity, y_gravity, z_gravity = np.array([]), np.array([]), np.array([])
    for i in Grav:
        x_gravity = np.append(x_gravity, i['x'])
        y_gravity = np.append(y_gravity, i['y'])
        z_gravity = np.append(z_gravity, i['z'])

    gravity = np.array([x_gravity, y_gravity, z_gravity])

    gravity_filt = LPfilter(gravity, fs) # LPF

    # # Plot gravity
    # plt.figure()
    # plt.plot(gravity_filt[0], label='x');
    # plt.plot(gravity_filt[1], label='y');
    # plt.plot(gravity_filt[2], label='z')
    # plt.xlim(0, len(gravity_filt[0]) - 1)
    # plt.title('Gravity')
    # plt.xlabel('Samples')
    # plt.legend()

    # Convert the quaternion vector to a rotation matrix using the following formula
    xq = q[0]; yq = q[1]; zq = q[2]; wq = q[3]
    R = np.array([
        [1 - 2 * yq * yq - 2 * zq * zq, 2 * xq * yq - 2 * wq * zq, 2 * xq * zq + 2 * wq * yq],
        [2 * xq * yq + 2 * wq * zq, 1 - 2 * xq * xq - 2 * zq * zq, 2 * yq * zq - 2 * wq * xq],
        [2 * xq * zq - 2 * wq * yq, 2 * yq * zq + 2 * wq * xq, 1 - 2 * xq * xq - 2 * yq * yq]
    ])

    # Multiply the rotation matrix by the gravity vector to obtain the acceleration due to gravity in the device's reference frame
    g_device = np.zeros((3, gravity_filt.shape[1]))
    for i in range(gravity_filt.shape[1]):
        g_device[:, i] = np.dot(R[:, :, i], gravity_filt[:, i])

    # Compute linear acceleration
    linear_acceleration = user_acceleration_filt - g_device

    # HF
    linear_acceleration_filt = HPfilter(linear_acceleration, fs)

    # # Plot linear acceleration
    # plt.figure()
    # plt.plot(linear_acceleration_filt[0], label='x');
    # plt.plot(linear_acceleration_filt[1], label='y');
    # plt.plot(linear_acceleration_filt[2], label='z')
    # plt.xlim(0, len(linear_acceleration_filt[0]) - 1)
    # plt.title('Linear acceleration')
    # plt.xlabel('Samples')
    # plt.legend()
    #
    # # Plot all figures
    # plt.show()

    # Standardize signals
    # Compute the mean and standard deviation for each axis
    linear_acceleration_filt_mean = np.mean(linear_acceleration_filt, axis=1)
    linear_acceleration_filt_std = np.std(linear_acceleration_filt, axis=1)

    # Standardize the data for each axis
    linear_acceleration_filt_standardized = (linear_acceleration_filt.T - linear_acceleration_filt_mean) / linear_acceleration_filt_std
    linear_acceleration_filt_standardized = linear_acceleration_filt_standardized.T

    # t = np.arange(0, linear_acceleration_filt_standardized.shape[1]) / fs
    # plt.plot(t, linear_acceleration_filt_standardized[0])
    # plt.plot(t, linear_acceleration_filt_standardized[1])
    # plt.plot(t, linear_acceleration_filt_standardized[2])
    # plt.xlabel('Time (s)')
    # plt.ylabel('Acceleration standardized (m/s^2)')
    # plt.show()

    return linear_acceleration_filt_standardized, fs

