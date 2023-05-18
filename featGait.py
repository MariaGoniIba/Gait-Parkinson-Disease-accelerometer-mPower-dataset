import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from matplotlib import pyplot as plt

def features_freqband(signal, fs, freq_band=None):
    # Function calculates frequential features. If no frequency band is specified, it calculates them over the whole signal
    if freq_band is None:
        # Perform FFT on the signal
        fft_result = np.fft.fft(signal)
        power = np.sum(np.abs(fft_result) ** 2) / len(signal)
        return power
    else:
        # Perform FFT on the signal
        fft_result = np.fft.fft(signal)

        # Calculate the corresponding frequency values
        freq_values = np.fft.fftfreq(len(signal), 1 / fs)

        # Find the indices of frequency values within the specified frequency band
        band_indices = np.where((freq_values >= freq_band[0]) & (freq_values <= freq_band[1]))[0]

        # Extract the amplitudes within the frequency band
        band_amplitudes = np.abs(fft_result[band_indices])

        # Find the index of the maximum amplitude within the frequency band
        max_amplitude_index = np.argmax(band_amplitudes)

        # Calculate the corresponding frequency of the maximum amplitude
        max_amplitude_freq = freq_values[band_indices][max_amplitude_index]

        # Calculate the energy and power within the frequency band
        energy = np.sum(np.abs(fft_result[band_indices]) ** 2)
        power = energy / len(band_indices)

        return power, band_amplitudes[max_amplitude_index], max_amplitude_freq


def features(acc, fs):
    # Compute the magnitude of the acceleration signal
    mag = np.sqrt(np.sum(acc ** 2, axis=0))

    # Find peaks
    minPeakHeight = 2 * np.std(mag)
    minPeakDistance = 80

    t = np.arange(0, acc.shape[1]) / fs

    peaks, _ = signal.find_peaks(mag, distance=80, height=2)
    # plt.plot(t, mag)
    # plt.plot(t[peaks], mag[peaks], 'ro')
    # plt.xlim(0, t[-1])
    # plt.show()

    # Number of steps
    # nSteps = len(peaks)

    ##### STRIDE BASED FEATURES #####
    # Mean stride interval (duration of a stride averaged over all strides)
    time_intervals = np.diff(peaks) / fs
    MSI = np.mean(time_intervals)
    # Stride variability (coefficient of variation)
    StrideVar = np.std(time_intervals) / np.mean(time_intervals)

    ##### Statistical features #####
    # Mean will be 0 and std 1 since I standardized so I dont include these variables.
    # Coefficient of variation (std/mean) also does not make sense here

    # Min
    min_x = np.min(acc[0]); min_y = np.min(acc[1]); min_z = np.min(acc[2]); min_mag = np.min(mag)

    # Max
    max_x = np.max(acc[0]); max_y = np.max(acc[1]); max_z = np.max(acc[2]); max_mag = np.max(mag)

    # Median
    median_x = np.median(acc[0]); median_y = np.median(acc[1]); median_z = np.median(acc[2]); median_mag = np.median(mag)

    # Interquartile range (q3 - q1)
    iqr_x = np.percentile(acc[0], 75) - np.percentile(acc[0], 25); iqr_y = np.percentile(acc[1], 75) - np.percentile(acc[1], 25)
    iqr_z = np.percentile(acc[2], 75) - np.percentile(acc[2], 25); iqr_mag = np.percentile(mag, 75) - np.percentile(mag, 25)

    # Skewness
    skew_x = skew(acc[0]); skew_y = skew(acc[1]); skew_z = skew(acc[2]); skew_mag = skew(mag)

    # Kurtosis
    kurtosis_x = kurtosis(acc[0]); kurtosis_y = kurtosis(acc[1]); kurtosis_z = kurtosis(acc[2]); kurtosis_mag = kurtosis(mag)

    # Zero-crossing rate
    # Find where the sign changes occur
    sign_changes_x = np.diff(np.sign(acc[0])); sign_changes_y = np.diff(np.sign(acc[1]))
    sign_changes_z = np.diff(np.sign(acc[2]));
    # Count the number of zero-crossings
    zcr_x = np.sum(sign_changes_x != 0); zcr_y = np.sum(sign_changes_y != 0)
    zcr_z = np.sum(sign_changes_z != 0);
    # Normalize by the signal length
    zcr_x /= len(acc[0]); zcr_y /= len(acc[1]); zcr_z /= len(acc[2]);

    ##### Frequential features #####
    # power of the whole signal
    power_x = features_freqband(acc[0], fs); power_y = features_freqband(acc[1], fs);
    power_z = features_freqband(acc[2], fs); power_mag = features_freqband(mag, fs)

    # Power, maximum amplitude and its frequency of occurrence in the locomotor band (0.5 - 3 Hz)
    power_LB_x, peak_LB_x, freq_peak_LB_x = features_freqband(acc[0], fs, (0.5, 3))
    power_LB_y, peak_LB_y, freq_peak_LB_y = features_freqband(acc[1], fs, (0.5, 3))
    power_LB_z, peak_LB_z, freq_peak_LB_z = features_freqband(acc[2], fs, (0.5, 3))
    power_LB_mag, peak_LB_mag, freq_peak_LB_mag = features_freqband(mag, fs, (0.5, 3))

    # Power, maximum amplitude and its frequency of occurrence in the freeze band (3 - 8 Hz)
    power_FB_x, peak_FB_x, freq_peak_FB_x = features_freqband(acc[0], fs, (3, 8))
    power_FB_y, peak_FB_y, freq_peak_FB_y = features_freqband(acc[1], fs, (3, 8))
    power_FB_z, peak_FB_z, freq_peak_FB_z = features_freqband(acc[2], fs, (3, 8))
    power_FB_mag, peak_FB_mag, freq_peak_FB_mag = features_freqband(mag, fs, (3, 8))

    # Freeze index
    FreezeInd_x = power_FB_x/power_LB_x; FreezeInd_y = power_FB_y/power_LB_y
    FreezeInd_z = power_FB_z / power_LB_z; FreezeInd_mag = power_FB_mag/power_LB_mag

    # RatioPower: sum of the power between freeze and locomotion band. To distinguish volitional standing from FoG
    RatioPower_x = power_FB_x + power_LB_x; RatioPower_y = power_FB_y + power_LB_y
    RatioPower_z = power_FB_z + power_LB_z; RatioPower_mag = power_FB_mag + power_LB_mag

    features = pd.DataFrame({'MSI': [MSI], 'StrideVar': [StrideVar],
                             'min_x': [min_x], 'min_y': [min_y], 'min_z': [min_z], 'min_mag': [min_mag],
                             'max_x': [max_x], 'max_y': [max_y], 'max_z': [max_z],'max_mag': [max_mag],
                             'median_x': [median_x],'median_y': [median_y],'median_z': [median_z],'median_mag': [median_mag],
                             'iqr_x': [iqr_x], 'iqr_y': [iqr_y], 'iqr_z': [iqr_z], 'iqr_mag': [iqr_mag],
                             'skew_x': [skew_x], 'skew_y': [skew_y], 'skew_z': [skew_z], 'skew_mag': [skew_mag],
                             'kurtosis_x': [kurtosis_x], 'kurtosis_y': [kurtosis_y], 'kurtosis_z': [kurtosis_z], 'kurtosis_mag': [kurtosis_mag],
                             'zcr_x': [zcr_x], 'zcr_y': [zcr_y], 'zcr_z': [zcr_z],
                             'power_x': [power_x],'power_y': [power_y], 'power_z': [power_z], 'power_mag': [power_mag],
                             'power_LB_x': [power_LB_x], 'power_LB_y': [power_LB_y], 'power_LB_z': [power_LB_z], 'power_LB_mag': [power_LB_mag],
                             'peak_LB_x': [peak_LB_x], 'peak_LB_y': [peak_LB_y], 'peak_LB_z': [peak_LB_z], 'peak_LB_mag': [peak_LB_mag],
                             'freq_peak_LB_x': [freq_peak_LB_x], 'freq_peak_LB_y': [freq_peak_LB_y], 'freq_peak_LB_z': [freq_peak_LB_z], 'freq_peak_LB_mag': [freq_peak_LB_mag],
                             'power_FB_x': [power_FB_x], 'power_FB_y': [power_FB_y], 'power_FB_z': [power_FB_z], 'power_FB_mag': [power_FB_mag],
                             'peak_FB_x': [peak_FB_x], 'peak_FB_y': [peak_FB_y], 'peak_FB_z': [peak_FB_z], 'peak_FB_mag': [peak_FB_mag],
                             'freq_peak_FB_x': [freq_peak_FB_x], 'freq_peak_FB_y': [freq_peak_FB_y], 'freq_peak_FB_z': [freq_peak_FB_z], 'freq_peak_FB_mag': [freq_peak_FB_mag],
                             'FreezeInd_x': [FreezeInd_x], 'FreezeInd_y': [FreezeInd_y], 'FreezeInd_z': [FreezeInd_z], 'FreezeInd_mag': [FreezeInd_mag],
                             'RatioPower_x': [RatioPower_x], 'RatioPower_y': [RatioPower_y], 'RatioPower_z': [RatioPower_z], 'RatioPower_mag': [RatioPower_mag]})
    return features