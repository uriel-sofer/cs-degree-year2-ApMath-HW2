import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time

def generate_cosine_wave(frequency, start_time, end_time):
    """
    Generates a cosine wave of given frequency between time range.
    :param frequency: Frequency of the cosine wave (Hz).
    :param start_time: Start time (seconds)
    :param end_time: End time (seconds)
    :return:
        t (numpy array): Time array.
        y (numpy array): Cosine wave values.
    """

    Ts = 1 / 10_000
    t = np.arange(start_time, end_time, Ts)
    y = np.cos(2 * np.pi * frequency * t)

    return t, y

def slice_11_cycles(t, y, frequency):
    """
    Slice the time and wave arrays to have only 11 cycles.
    :param frequency: Frequency of the cosine wave (Hz).
    :param t: Time array.
    :param y: Cosine wave values.
    :return:
        t_11 (numpy array): Time array for 11 cycles.
        y_11 (numpy array): Cosine wave values for 11 cycles.
    """

    T = 1 / frequency
    samples_needed = int(11 * T * 10_000)
    t_11 = t[:samples_needed]
    y_11 = y[:samples_needed]
    return t_11, y_11

def generate_cosine_series():
    """
    Generates a cosine series from 500Hz to 20,000Hz with steps of 500, then plays and plots.
    """

    fs = 10_000
    start_time = 0
    end_time = 1

    for f0 in range(500, 20_001, 500):
        t, x = generate_cosine_wave(f0, start_time, end_time)

        sd.play(x, fs)

        plt.plot(t[:100], x[:100])
        plt.title(f"Cosine wave of {f0} Hz")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.grid()
        plt.show()

        sd.wait()

def generate_chromatic_scale():
    """
    Generate and play the chromatic scale from a 440Hz to a 20,000Hz.
    """
    base_freq = 440
    max_freq = 20_000
    n = 0

    while True:
        freq = base_freq * (2 ** (n / 12))
        if freq > max_freq:
            break

        t, y = generate_cosine_wave(freq, 0, 0.5)

        plt.plot(t[:100], y[:100])
        plt.title(f"Cosine wave of {freq: .2f} Hz")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.grid()
        plt.show()

        print(f"Playing {freq: .2f} Hz")
        sd.play(y, 10_000)
        sd.wait()

        n += 1

def play_specific_notes(frequencies_lst, durations_lst):
    """
    Play the specified notes over the specified durations.
    :param frequencies_lst:
    :param durations_lst:
    """

    for frequency, duration in zip(frequencies_lst, durations_lst):
        if frequency == 0:
            print(f"Silence for {duration: .2f} seconds")
            sd.sleep(int(duration * 1000))

        else:
            print(f"Playing {frequency: .2f} Hz for {duration: .2f} seconds")
            t, y = generate_cosine_wave(frequency, 0, duration)
            sd.play(y, 10_000)
            sd.wait()

frequencies = [
 784, 880, 988, 1047, 988, 880, 784, 988, 880, 784, 0,
 784, 880, 988, 1047, 988, 880, 784, 988, 880, 784, 0,
 880, 988, 1047, 1175, 1047, 988, 880, 784, 880, 988, 0,
 988, 880, 784, 988, 880, 784, 0
]

durations = [ 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25,
0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25,
0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.25 ]

generate_cosine_series()
generate_chromatic_scale()
play_specific_notes(frequencies, durations)