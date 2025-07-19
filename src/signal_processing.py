import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from Global_Translator import translator

def zcr(signal):
    sig_cross = []

    for i in range(len(signal) - 1):
        cross = 0.5 * (np.abs(np.sign(signal[i]) - np.sign(signal[i + 1])))

        if not cross:
            sig_cross.append(0)
        else:
            sig_cross.append(1)

    return sig_cross


def median_filter(signal, neighs=10):
    sig = signal.copy()
    for i in range(neighs // 2):
        sig[i] = np.median(signal[:i + (neighs // 2)])

    for i in range(neighs // 2, len(sig) - (neighs // 2)):
        sig[i] = np.median(signal[i - (neighs // 2):i + (neighs // 2)])

    for i in range(len(sig) - (neighs // 2), len(sig)):
        sig[i] = np.median(signal[i - (neighs // 2):])

    return np.array(sig)


def mean_filter(signal, neighs=10):
    sig = signal
    for i in range(neighs // 2):
        sig[i] = np.mean(signal[:i + (neighs // 2)])

    for i in range(neighs // 2, len(sig) - (neighs // 2)):
        sig[i] = np.mean(signal[i - (neighs // 2):i + (neighs // 2)])

    for i in range(len(sig) - (neighs // 2), len(sig)):
        sig[i] = np.mean(signal[i - (neighs // 2):])

    return np.array(sig)


def list_signals(folder_path):
    """Returns a list of all .txt and .wav files in the given folder and subfolders."""
    import os
    files = []
    for root, _, filenames in os.walk(folder_path):
        files.extend(os.path.join(root, f) for f in filenames if f.endswith(('.txt', '.wav', '.csv')))
    return files


class TS:
    def __init__(self, input_data, sr=None, name_regex=False, CompanyName=''):
        self.statistical_features = None
        self.extra_information = {}
        import pandas as pd

        if isinstance(input_data, (list, np.ndarray)):  # Check if input is a list of numbers
            signal = np.array(input_data)
            if sr is None:
                raise ValueError("Sample rate (sr) must be provided for time signal list input.")
            length = len(signal) / sr
            details = self.create_details(len(signal), length)

        elif isinstance(input_data, str) and ((input_data[-3:]).lower() == 'wav'):  # Check if input is a .wav file
            import librosa
            signal, sr = librosa.load(input_data)
            length = len(signal) / sr
            details = self.create_details(len(signal), length)

        elif isinstance(input_data, str) and ((input_data[-3:]).lower() == 'txt'):  # Assume input is a CSV path
            import os

            signal = pd.read_csv(input_data, header=None, encoding="latin1")
            details = signal[:9]
            signal = np.asarray(signal[9:].reset_index(drop=True)).flatten().astype('float')
            length_str = str(details[0][6])
            length = float(length_str[10:])

            self.name = os.path.basename(input_data)

            if name_regex:
                reg_dict = translator(input_data, CompanyName)
                self.comp_number, self.point, self.direction, self.assignment, self.date, self.time, self.spare = reg_dict['name'], reg_dict['point'], reg_dict['direction'], reg_dict['assignment'], reg_dict['date'], reg_dict['hour'], reg_dict['spare_part']

        records_str = str(details[0][2])
        records = int(records_str[14:])

        self.file_path = input_data
        self.signal = signal
        self.details = details
        self.length = length
        self.sample_rate = records / length

        self.fq = []
        self.ft = []
        self.x_vals = []
        self.x_args = []

        self.frames = []
        self.env = []
        self.stft = []
        self.theta = []
        self.circular_x = []
        self.circular_y = []

        self.frame_size = 512
        self.hop_length = 256
        self.rpm = None

        self.resampled_signal = []
        self.resampled_signal_sr = 0

        self.has_half_1X = False

    def set_extra_information(self, key, value):
        self.extra_information[key] = value

    def run_translator(self):
        import re
        from datetime import datetime
        try:
            signal_address = self.name.lower().replace(' - Time Signal.txt', '')
            og = signal_address
            output = {
                "machine_name": 'None',
                "point": 'None',
                "direction": 'None',
                "spare_part": 'None',
                "assignment": 'None',
                "date": 'None',
                "time": 'None',
                "component_id": 'None'
            }

            # Extract assignment
            assignment_match = re.search(r"(vel|acc|hd|spm)", signal_address)
            if not assignment_match:
                print("Invalid signal format: Missing assignment")
                print(og)
                output["assignment"] = 'None'
            else:
                output["assignment"] = assignment_match.group(1).capitalize()
                signal_address = signal_address.replace(assignment_match.group(1), '')

            # Extract date
            date_match = re.search(r"\((\d+_\d+_\d+ \d+_\d+_\d+)\)", signal_address)
            if not date_match:
                print("Invalid signal format: Missing date")
                output["date"] = 'None'
                output["time"] = 'None'
            else:
                datetime_str = date_match.group(1)
                dt = datetime.strptime(datetime_str, "%m_%d_%Y %H_%M_%S")
                output["date"] = dt.strftime("%Y-%m-%d")  # Date as string
                output["time"] = dt.strftime("%H:%M:%S")  # Time as string
                signal_address = signal_address.replace(date_match.group(1), '')

            # Extract machine name
            machine_name_with_extra = signal_address.split(' ')[0]
            if not machine_name_with_extra:
                print("Invalid signal format: Missing machine name")
                print(og)
                output["machine_name"] = 'None'
            else:
                machine_name = re.sub(r'([-.]\d*[ahv]\d*|\.\d+)$', '', machine_name_with_extra)

                output["machine_name"] = machine_name.upper()
                signal_address = signal_address.replace(machine_name_with_extra, '')

            # Extract component id (not important, for testing purposes)
            component_match_1 = re.search(r"((m|f|g|p|r|mill|pinion)-?\d+-?([ahv])(\d+)?_?)", signal_address)
            component_match_2 = re.search(r"((m|f|g|p|r|mill|pinion)-?\d+-?([ahv])?(\d+)?_?)", signal_address)
            component_match_3 = re.search(r"((m|f|g|p|r|mill|pinion)?-?\d+-?([ahv])(\d+)?_?)", signal_address)
            component_match_4 = re.search(r"((m|f|g|p|r|mill|pinion)?-?\d+-?([ahv])?(\d+)?_?)", signal_address)
            best_component_match = component_match_1 if component_match_1 else component_match_2 if component_match_2 else component_match_3 if component_match_3 else component_match_4
            if not best_component_match:
                print("Invalid signal format: Missing component")
                print(og)
                component_id = 'None'
                output["component_id"] = 'None'
                output["spare_part"] = 'None'
                output["direction"] = 'None'
                output["point"] = 'None'
            else:
                component_id = best_component_match.group(
                    0)
                output["component_id"] = component_id
                spare_part_map = {'mill': 'Mill', 'pinion': 'Pinion',
                                  'm': 'Electro Motor', 'f': 'Fan', 'g': 'Gearbox',
                                  'p': 'Pump', 'r': 'Rotor'
                                  }
                output["spare_part"] = next((v for k, v in spare_part_map.items() if k in component_id), 'None')
                direction_map = {'h': 'H', 'a': 'A', 'v': 'V'}
                output["direction"] = next((v for k, v in direction_map.items() if k in component_id), 'None')
                point_match = re.search(r"\d+", component_id)
                if not point_match:
                    print("Invalid signal format: Missing point")
                    print(og)
                    output["point"] = 'None'
                else:
                    output["point"] = str(point_match.group(0))

            self.translator_data = output

        except Exception as e:
            print(f"Error: {e}")
            return None

    def extract_statistical_features(self):
        import scipy.stats as stats
        X = np.array(self.signal)

        # Time domain features
        features = {
            'min': np.min(X),
            'max': np.max(X),
            'mean': np.mean(X),
            'rms': np.sqrt(np.mean(X ** 2)),
            'var': np.var(X),
            'std': np.std(X),
            'power': np.sum(X ** 2) / len(X),
            'peak': np.max(np.abs(X)),
            'peak_to_peak': np.ptp(X),
            'crest_factor': np.max(np.abs(X)) / np.sqrt(np.mean(X ** 2)),
            'skew': stats.skew(X),
            'kurtosis': stats.kurtosis(X),
            'form_factor': np.sqrt(np.mean(X ** 2)) / np.mean(X),
            'pulse_indicator': np.max(np.abs(X)) / np.mean(np.abs(X)),
            'margin_factor': np.max(np.abs(X)) / np.mean(np.abs(X)) ** 2,
            'shape_factor': np.sqrt(np.max(X ** 2)) / np.mean(np.abs(X)),
            'impulse_factor': np.max(np.abs(X)) / np.mean(np.abs(X)),
            'energy': np.sum(X ** 2),
        }

        # Frequency domain features

        freqs, ffts = self.fq, self.ft
        features.update({
            'max_f': np.max(ffts),
            'sum_f': np.sum(ffts),
            'mean_f': np.mean(ffts),
            'var_f': np.var(ffts),
            'crest_factor_f': np.max(np.abs(ffts)) / np.sqrt(np.mean(ffts ** 2)),
            'skew_f': stats.skew(ffts),
            'kurtosis_f': stats.kurtosis(ffts),
            'peak_to_peak_f': np.ptp(ffts)
        })

        self.statistical_features = features

    def create_details(self, num_items, length):
        import pandas as pd

        type = '% Type: List'
        date = '% Date: '
        items = f'% NoOfItems: = {num_items}'
        unit_y = '% Unit_Y: = g'
        unit_x = '% Unit_X: = Sec'
        min_x = '% Min_X: = 0'
        max_x = f'% Max_X: = {length}'
        min_x_dis = '% MinDisplay_X: = 0'
        max_x_dis = f'% MaxDisplay_X: = {length}'
        details = pd.DataFrame([type, date, items, unit_y, unit_x, min_x, max_x, min_x_dis, max_x_dis])
        return details

    def rms(self) -> float:
        """
        Calculate the Root Mean Square (RMS) value of the signal.

        Returns:
        float: The RMS value, rounded to 5 decimal places.
        """
        return round(np.sqrt(np.mean(self.signal ** 2)), 5)

    def peak(self) -> float:
        """
        Calculate the peak (maximum absolute) value of the signal.

        Returns:
        float: The peak value, rounded to 5 decimal places.
        """
        return round(np.max(np.abs(self.signal)), 5)

    def peak_to_peak(self) -> float:
        """
        Calculate the peak-to-peak value of the signal.

        Returns:
        float: The difference between the maximum and minimum values, rounded to 5 decimal places.
        """
        return round(np.max(self.signal) - np.min(self.signal), 5)

    def crest(self) -> float:
        """
        Calculate the crest factor of the signal.
        Crest Factor = Peak Value / RMS Value

        Returns:
        float: The crest factor, rounded to 5 decimal places.
        """
        rms_value = self.rms()
        return round(self.peak() / rms_value, 5) if rms_value != 0 else float('inf')

    def kurt(self) -> float:
        """
        Calculate the kurtosis of the signal, which measures the tailedness of the distribution.

        Returns:
        float: The kurtosis value, rounded to 5 decimal places.
        """
        from scipy.stats import kurtosis

        return round(kurtosis(self.signal), 5)

    def shape_factor(self) -> float:
        """
        Calculate the shape factor of the signal, which is the ratio of the RMS value to 
        the mean of the absolute values. It gives an indication of the waveform shape.

        Returns:
        float: The shape factor value, rounded to 5 decimal places.
        """
        rms = self.rms()
        mean_abs = self.absolute_mean()
        return round(rms / mean_abs, 5)

    def skewness(self) -> float:
        """
        Calculate the skewness of the signal, which measures asymmetry in the distribution.

        Returns:
        float: The skewness value, rounded to 5 decimal places.
        """
        from scipy.stats import skew
        return round(skew(self.signal), 5)

    def entropy(self) -> float:
        """
        Calculate the entropy of the signal.

        Returns:
        float: The entropy value, rounded to 5 decimal places.
        """
        from scipy.stats import entropy

        # Compute probability distribution of signal values
        hist, bin_edges = np.histogram(self.signal, bins=50, density=True)
        probabilities = hist / np.sum(hist)

        # Shannon entropy calculation
        shannon_entropy = entropy(probabilities, base=2)
        return round(shannon_entropy, 5)

    def zcr(self) -> int:
        """
        Calculate the Zero Crossing Rate (ZCR) of the signal.
        ZCR is the number of times the signal crosses the zero level.

        Returns:
        int: The number of zero crossings in the signal.
        """
        sig_cross = []
        for i in range(len(self.signal) - 1):
            cross = 0.5 * (np.abs(np.sign(self.signal[i]) - np.sign(self.signal[i + 1])))
            if not cross:
                sig_cross.append(0)
            else:
                sig_cross.append(1)
        return sum(sig_cross)

    def impulse_factor(self) -> float:
        """
        Calculate the Impulse Factor.
        Impulse Factor = Peak Value / Mean Value

        Returns:
        float: The impulse factor, rounded to 5 decimal places.
        """
        mean_value = np.mean(self.signal)
        return round(self.peak() / mean_value, 5) if mean_value != 0 else float('inf')

    def clearance_factor(self) -> float:
        """
        Calculate the Clearance Factor.
        Clearance Factor = Peak Value / (Mean of sqrt(abs(signal)))^2

        Returns:
        float: The clearance factor, rounded to 5 decimal places.
        """
        mean_sqrt_abs = np.mean(np.sqrt(np.abs(self.signal)))
        return round(self.peak() / (mean_sqrt_abs ** 2), 5) if mean_sqrt_abs != 0 else float('inf')

    def absolute_mean(self) -> float:
        """
        Calculate the Absolute Mean of the signal.

        Returns:
        float: The mean of the absolute values, rounded to 5 decimal places.
        """
        return round(np.mean(np.abs(self.signal)), 5)

    def standard_deviation(self) -> float:
        """
        Calculate the Standard Deviation of the signal.

        Returns:
        float: The standard deviation, rounded to 5 decimal places.
        """
        return round(np.std(self.signal), 5)

    def framing(self):
        frames = []
        for i in range(0, len(self.signal) - self.hop_length, self.hop_length):
            current_frame = self.signal[i:i + self.frame_size]
            frames.append(current_frame)
        self.frames = frames

    def amplitude_env(self):
        """
        Calculate the amplitude envelope of time signal.

        Modifies:
        self.env (np.ndarray): Computed envelope of time signal.
        
        Returns:
        None
        """
        from scipy.signal import detrend, hilbert

        analytic_signal = hilbert(self.signal)
        envelope = np.abs(analytic_signal)

        envelope = detrend(envelope)
        self.env = envelope

    def play_audio(self):
        """
        Play time signal's audio (velocity or acceleration).
        """
        from IPython.display import Audio
        if self.peak() > 1:
            aud = self.signal / self.peak()
        else:
            aud = self.signal

        return Audio(aud, rate=int(self.sample_rate))

    def fftransform(self):
        """
        Computes Fast Fourier Transform (FFT) of time signal.
        
        Modifies:
        FFT-amplitude in self.ft and FFT-frequency in self.fq
        """

        N = len(self.signal)
        T = 1 / self.sample_rate
        ft = np.fft.fft(self.signal)
        freq = np.fft.fftfreq(N, T)[:N // 2]

        transformed_signal = 1.44 * (np.abs(ft[0:N // 2])) / N

        self.ft = transformed_signal
        self.fq = freq

    def spectral_entropy(self):

        power_spectrum = np.abs(self.ft) ** 2
        power_spectrum /= np.sum(power_spectrum)

        power_spectrum = np.where(power_spectrum == 0, 1e-12, power_spectrum)
        entropy = -np.sum(power_spectrum * np.log2(power_spectrum))

        return entropy

    def stftransform(self, n_fft=512, hop_length=256, window="hann", freq_range=None):
        """
        Computes the STFT of the framed signal using Librosa.

        Parameters:
        - n_fft: Frame size (e.g., 1024 for good frequency resolution)
        - hop_length: Step size (e.g., 256 for 75% overlap)
        - window: Window function (e.g., "hann" for smooth transitions)
        - freq_range: Tuple (min_freq, max_freq) in Hz to select frequency range. If None, uses full spectrum.
        """
        import librosa

        self.framing()
        signal = np.concatenate(self.frames, axis=0)

        # Compute STFT
        stft_matrix = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, window=window)
        magnitude = np.abs(stft_matrix)

        # If freq_range specified, filter frequencies
        if freq_range is not None:
            sr = self.sample_rate  # assuming you have sampling_rate attribute
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            min_freq, max_freq = freq_range

            # Find indices within range
            freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
            magnitude = magnitude[freq_mask, :]
            freqs = freqs[freq_mask]

            self.freqs = freqs  # Save filtered frequencies
        else:
            self.freqs = librosa.fft_frequencies(sr=self.sampling_rate, n_fft=n_fft)

        self.stft = magnitude

    def stftransform_2(self, nperseg=1024, noverlap=512):
        from scipy import signal
        f, t_stft, Zxx = signal.stft(self.signal, fs=self.sample_rate, nperseg=nperseg, noverlap=noverlap)
        self.stft = f, t_stft, Zxx

    def get_length(self):
        length_str = str(self.details[0][6])
        length = float(length_str[10:])

        return length

    def resample(self, sample_rate, new_sample_rate):
        from scipy.signal import resample_poly

        resampled_signal = resample_poly(self.signal, new_sample_rate, int(sample_rate))
        return resampled_signal

    def acc_to_vel(self, init_vel=0, degree=2):
        """
        Computes velocity from acceleration using cumulative trapzoid.
        
        Returns: Velocity array.
        """
        from scipy.integrate import cumulative_trapezoid

        # FIXME: add low-pass later. maybe worked better.

        time_unit = 1 / self.sample_rate
        num_points = len(self.signal)

        max_time = num_points * time_unit
        time = np.linspace(0, max_time - time_unit, num_points)

        signal_copy = self.signal
        signal_copy = signal_copy - np.mean(signal_copy)

        velocity = cumulative_trapezoid(signal_copy, time, initial=init_vel)
        if degree != 0:
            p = np.polyfit(time, velocity, deg=degree)  # Fit quadratic trend
            velocity -= np.polyval(p, time)  # Subtract trend
        else:
            from scipy.signal import detrend
            velocity = detrend(velocity)

        return velocity * 986

    def vel_to_acc(self):
        """
        Computes acceleration from velocity using numerical differentiation.
        
        Returns:
        Acceleration array.
        """
        dt = 1 / self.sample_rate  # Time step
        acceleration = np.diff(self.signal) / dt  # Numerical differentiation
        acceleration = np.append(acceleration, acceleration[-1])  # Maintain same length
        return acceleration

    def circular_theta(self):
        theta = []
        for i in range(int(self.sample_rate / (self.rpm / 60))):
            theta.append(360 * i / int(self.sample_rate / (self.rpm / 60)))
        self.theta = theta

    def circular(self, rounds=1):

        self.circular_theta()

        xs = []
        ys = {}

        theta_radians = np.deg2rad(self.theta)
        round_idx = 1
        for i in range(rounds):
            xs.append(np.round(theta_radians, 3))
            start_idx = int(self.sample_rate / (self.rpm / 60)) * i
            end_idx = int(self.sample_rate / (self.rpm / 60)) * (i + 1)
            ys[f'round{round_idx}'] = (np.round(self.signal[start_idx:end_idx], 2)).tolist()

            round_idx += 1
        self.circular_x, self.circular_y = np.round(xs[0],2), ys

    def circular_plt(self, rounds=1):

        self.circular_theta()
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))

        theta_radians = np.deg2rad(self.theta)

        for i in range(rounds):
            ax.plot(theta_radians, self.signal[int(self.sample_rate / (self.rpm / 60)) * i: int(
                self.sample_rate / (self.rpm / 60)) * (i + 1)], 'blue')

        ax.set_rmin(min(self.signal))
        ax.set_rmax(max(self.signal))
        ax.set_title(f'Circular Time Signal - {rounds} rounds')
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        plt.show()

    def fft_plt(self, xlimit=1000, ylim=None, grd=True, linewidth='0.7', title=None, show_harmonic_peaks=False):
        plt.figure(figsize=(30, 7))
        plt.xticks(np.arange(0, xlimit, xlimit / 50))
        plt.xlim(0, xlimit)
        plt.ylim(min(self.ft), max(self.ft[1:]) + (max(self.ft[1:]) / 10))
        plt.plot(self.fq, self.ft, linewidth=linewidth)
        plt.fill_between(self.fq, self.ft, color='skyblue', alpha=0.7)

        pltymin, pltymax = plt.ylim()

        if (len(self.x_args) != 0) and (len(self.x_vals) != 0):
            for i, (x_freq, x_val) in enumerate(zip(self.x_args, self.x_vals)):
                plt.vlines(x=x_freq, ymin=x_val, ymax=pltymax, color='red', linestyle='-.', linewidth=0.8, alpha=0.8)
                y_mid = (x_val + pltymax) / 2.5

                # Calculate the X multiplier label (e.g., 0.5X, 1.5X, etc.)
                if self.rpm:
                    one_x_freq = self.rpm / 60
                else:
                    one_x_freq = self.get_1x_freq(50)

                multiplier = round(x_freq / one_x_freq, 1)

                plt.text(x_freq, y_mid, f'{multiplier}X RPM freq: {x_freq:0.1f}', color='red', rotation=90,
                         verticalalignment='center', horizontalalignment='right')
        if show_harmonic_peaks:
            # Show red square markers on harmonic peaks
            plt.plot(self.x_args, self.x_vals, 's', color='red', label='Harmonic Peaks')

        if ylim is not None:
            plt.ylim(min(self.ft), ylim)

        plt.grid(alpha=0.1)
        plt.grid(grd)

        if title:
            plt.text(0.01 * xlimit, pltymax * 0.92, title, fontsize=20, fontweight='bold',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

        plt.show()

    def fft_plt_interactive(self, xlimit=1000, ylim=None, linewidth=0.7):
        import plotly.graph_objects as go

        fig = go.Figure()

        # Main FFT line
        fig.add_trace(go.Scatter(
            x=self.fq,
            y=self.ft,
            mode='lines',
            line=dict(width=linewidth, color='blue'),
            name='FFT'
        ))

        # Fill under the curve
        fig.add_trace(go.Scatter(
            x=self.fq,
            y=self.ft,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(135, 206, 235, 0.5)',  # Light blue
            line=dict(width=0),
            name='Amplitude'
        ))

        # Optional markers
        if len(self.x_args) > 0 and len(self.x_vals) > 0:
            for idx in range(len(self.x_args)):
                fig.add_trace(go.Scatter(
                    x=[self.x_args[idx]],
                    y=[self.x_vals[idx]],
                    mode='markers+text',
                    text=[f'{idx + 1}X RPM freq: {self.x_args[idx]:0.1f}'],
                    textposition="top center",
                    marker=dict(color='red', size=8),
                    name=f'{idx + 1}X RPM Marker'
                ))

        # Set axis limits
        fig.update_layout(
            xaxis=dict(range=[0, xlimit], title='Frequency (Hz)', tickvals=np.arange(0, xlimit, xlimit / 20)),
            yaxis=dict(range=[min(self.ft), ylim if ylim else max(self.ft) + (max(self.ft) / 10)], title='Amplitude'),
            title='Interactive FFT Plot',
            template='plotly_white'
        )

        # Show grid and interactivity
        fig.update_layout(showlegend=True)
        fig.show()

    def sig_plt(self, col='red', width='1', fraction=None, grd=True, title=''):
        plt.figure(figsize=(30, 7))

        if fraction is None:
            plot_signal = self.signal
            plot_time = np.linspace(0, len(self.signal) * (1 / self.sample_rate), len(self.signal))

        else:
            plot_signal = self.signal[fraction[0]:fraction[1]]
            plot_time = np.linspace(0, len(plot_signal) * (1 / self.sample_rate), len(plot_signal))

        plt.plot(plot_time, plot_signal, color=col, linewidth=width)

        max_x_value = len(plot_signal) / self.sample_rate
        x_ticks = np.round(np.linspace(0, max_x_value, 10), 3)

        plt.xticks(x_ticks, rotation=90)
        plt.title(title)

        if grd:
            plt.grid(True, alpha=0.3)

        plt.xlabel('Time (s)', fontsize=15)
        plt.ylabel('Amplitude', fontsize=15)

        plt.show()

    def sig_env_plt(self):
        plt.figure(figsize=(30, 7))

        env_x = np.arange(0, len(self.signal), len(self.signal) / len(self.env))
        plt.plot(env_x, self.env, color='red', linewidth=1.5)
        plt.plot(self.signal, color='blue', alpha=0.5)

        plt.xticks(np.arange(0, len(self.signal) + len(self.signal) / 20, len(self.signal) / 20), rotation=90)
        plt.grid(True, linestyle='-.')

        plt.show()

    def stft_plt(self):
        plt.figure(figsize=(30, 7))
        plt.imshow(self.stft, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Spectrogram')
        plt.show()

    def find_harmonics(self):

        harmonics_amp = self.x_vals
        fft_vals = self.ft

        ignore = False

        _1x = harmonics_amp[0]
        noise_level = np.mean(fft_vals)
        print(noise_level * 50)
        if _1x < noise_level * 50:
            return -1  # the signal is healthy and doesn't have harmonics

        idx = 0
        xs = 0

        while not ignore and idx < len(harmonics_amp):

            if ((harmonics_amp[idx] / _1x) * 100) < 8:

                if (idx != (len(harmonics_amp) - 1)) and ((harmonics_amp[idx + 1] / _1x) * 100) > 8:
                    xs = idx + 1
                else:
                    ignore = True

            else:
                xs = idx + 1
            idx += 1

        return xs

    def get_1x_freq(self, denoise_fact=200):

        # fix - small difference: rpm = estimated_rpm
        # fix - large difference: rpm = estimated_rpm in history

        # var - small difference: rpm = estimated_rpm
        # var - large difference: rpm = estimated_rpm in other points

        # high rpm: 1Hz
        # low rpm: 0.1-0.5Hz ~ 10RPM
        # low rpm (<60): HD-Env - fft-max

        # 3111FN1, 3218FN1, 3618FN1
        # Gearbox: Gmf/user_RPM: integer(teeth) -> correct user_RPM

        denoised_signal = TS(mean_filter(self.signal, denoise_fact), self.sample_rate)
        denoised_signal.fftransform()
        return denoised_signal.fq[np.argmax(denoised_signal.ft)]

    def harmonic_detect(self):

        for idx in range(len(self.fq)):
            if self.x_args[0] - 1 < int(self.fq[idx]) < self.x_args[0] + 1:
                _1x_idx = idx
            if self.x_args[1] - 1 < int(self.fq[idx]) < self.x_args[1] + 1:
                _2x_idx = idx
                break

        search_rng = int(10 // self.fq[1])
        _1x_back_noise = (np.mean(self.ft[max(0, _1x_idx - search_rng):_1x_idx + search_rng]) * 4)
        _2x_back_noise = (np.mean(self.ft[max(0, _2x_idx - search_rng):_2x_idx + search_rng]) * 3)

        if ((
                self.x_vals[0] < _1x_back_noise
        ) or (
                self.x_vals[1] < _2x_back_noise
        )):
            return -1

        ignore = False

        _1x = self.x_vals[0]

        if (_1x < (0.1 * np.max(self.ft))):
            return -1  # the signal is healthy and doesn't have harmonics

        idx = 0
        xs = 0

        while not ignore and idx < len(self.x_vals):

            if ((self.x_vals[idx] / _1x) * 100) < 8:

                if (idx != (len(self.x_vals) - 1)) and ((self.x_vals[idx + 1] / _1x) * 100) > 8:
                    xs = idx + 1
                else:
                    ignore = True

            else:
                xs = idx + 1
            idx += 1

        return xs

    # def get_xs_freqs(self, xs: int, dynamic_peaks: Literal['super-strict', 'strict', 'normal', 'dynamic']=None, need_half_1X=False):
    #     '''
    #     #### NOTE: First set the signal.rpm to machine's RPM value
    #     This function computes harmonics of time signal;
    #     Saves frequency of harmonics in x_args and amplitudes in x_vals.

    #     Arguments:
    #         xs: number of harmonics you need
    #         dynamic_peaks: 'super-strict' (0 idx around), 'strict' (2 idx around), 'normal' (5 idx around), 'dynamic' (15 idx around)
    #     '''
    #     x_args = []
    #     x_vals = []
    #     x_idxs = []

    #     _1x_freq = self.get_1x_freq(50)

    #     if self.rpm:
    #         sample_for_round_float = (self.rpm / 60) / self.fq[1]
    #     else:
    #         sample_for_round_float = _1x_freq / self.fq[1]

    #     if not dynamic_peaks:
    #         if _1x_freq < 10:
    #             dynamic_peaks = 'super-strict'
    #         elif 10<=_1x_freq<15:
    #             dynamic_peaks = 'strict'
    #         else:
    #             dynamic_peaks = 'normal'

    #     search_range = 5
    #     if dynamic_peaks=='dynamic':
    #         search_range = 15
    #     elif dynamic_peaks=='strict':
    #         search_range = 2
    #     elif dynamic_peaks=='super-strict':
    #         search_range = 0    

    #     for x in range(1,xs+1):            

    #         samples_this_round = int(x*sample_for_round_float)
    #         if dynamic_peaks!='super-strict':

    #             harmonic_idx = (np.argmax(self.ft[max(0, samples_this_round-search_range):samples_this_round+search_range]))+max(0, samples_this_round-search_range)

    #             x_val = round(self.ft[harmonic_idx],3)
    #             x_arg = self.fq[harmonic_idx]
    #             x_idx = harmonic_idx

    #         else:
    #             x_val = round(self.ft[samples_this_round],3)
    #             x_arg = self.fq[samples_this_round]
    #             x_idx = samples_this_round

    #         x_args.append(x_arg)
    #         x_vals.append(x_val)
    #         x_idxs.append(x_idx)

    #     if need_half_1X:
    #         self.has_half_1X = True            
    #         arnd_half_1X = int(sample_for_round_float/2)           
    #         if len(self.ft[arnd_half_1X-2:arnd_half_1X+2]):
    #             half_1X = np.argmax(self.ft[arnd_half_1X-2:arnd_half_1X+2]) + (arnd_half_1X-2)

    #             x_args.insert(0, self.fq[half_1X])
    #             x_vals.insert(0, self.ft[half_1X])          
    #             x_idxs.insert(0, half_1X)

    #     self.x_args, self.x_vals, self.x_idxs = np.round(x_args,4).tolist(), np.round(x_vals,4).tolist(), x_idxs
    def get_xs_freqs(self, xs: int, dynamic_peaks: Literal['super-strict', 'strict', 'normal', 'dynamic'] = None,
                     need_half_1X=False):
        '''
        #### NOTE: First set the signal.rpm to machine's RPM value
        This function computes harmonics of time signal;
        Saves frequency of harmonics in x_args and amplitudes in x_vals.

        Arguments:
            xs: number of harmonics you need
            dynamic_peaks: 'super-strict' (0 idx around), 'strict' (2 idx around), 'normal' (5 idx around), 'dynamic' (15 idx around)
        '''
        x_args = []
        x_vals = []
        x_idxs = []


        if self.rpm:
            _1x_freq = self.rpm / 60
            sample_for_round_float = (self.rpm / 60) / self.fq[1]
        else:
            _1x_freq = self.get_1x_freq(50)
            sample_for_round_float = _1x_freq / self.fq[1]

        if not dynamic_peaks:
            if _1x_freq < 10:
                dynamic_peaks = 'super-strict'
            elif 10 <= _1x_freq < 15:
                dynamic_peaks = 'strict'
            else:
                dynamic_peaks = 'normal'

        search_range = 5
        if dynamic_peaks == 'dynamic':
            search_range = 15
        elif dynamic_peaks == 'strict':
            search_range = 2
        elif dynamic_peaks == 'super-strict':
            search_range = 0

        for x in range(1, xs + 1):

            if self.fq[-1] < (_1x_freq * x):
                break

            samples_this_round = int(x * sample_for_round_float)
            if dynamic_peaks != 'super-strict':

                harmonic_idx = (np.argmax(
                    self.ft[max(0, samples_this_round - search_range):samples_this_round + search_range])) + max(0,
                                                                                                                 samples_this_round - search_range)

                x_val = round(self.ft[harmonic_idx], 3)
                x_arg = self.fq[harmonic_idx]
                x_idx = harmonic_idx

            else:
                x_val = round(self.ft[samples_this_round], 3)
                x_arg = self.fq[samples_this_round]
                x_idx = samples_this_round

            x_args.append(x_arg)
            x_vals.append(x_val)
            x_idxs.append(x_idx)

        if need_half_1X:
            self.has_half_1X = True
            arnd_half_1X = int(sample_for_round_float / 2)
            if len(self.ft[arnd_half_1X - 2:arnd_half_1X + 2]):
                half_1X = np.argmax(self.ft[arnd_half_1X - 2:arnd_half_1X + 2]) + (arnd_half_1X - 2)

                x_args.insert(0, self.fq[half_1X])
                x_vals.insert(0, self.ft[half_1X])
                x_idxs.insert(0, half_1X)

        self.x_args, self.x_vals, self.x_idxs = np.round(x_args, 4).tolist(), np.round(x_vals, 4).tolist(), x_idxs

    def get_xs_freqs_and_validate(self, xs, alpha, action):
        self.fftransform()
        self.get_xs_freqs(xs=xs)
        harmonics = np.array(self.x_vals[:10])
        avg_harmonic = np.mean(harmonics)
        max_harmonic_limit = avg_harmonic + alpha * avg_harmonic
        if action == 'limit':
            self.x_vals = [min(j, max_harmonic_limit) for j in self.x_vals]
        elif action == 'zero':
            self.x_vals = [0 if j > max_harmonic_limit else j for j in self.x_vals]
        elif action == 'half':
            self.x_vals = [j / 2 if j > max_harmonic_limit else j for j in self.x_vals]


    def hd_envelope(self, lowcut=4500, highcut=8000, neighbors=50):
        filtsig = TS(self.bandpass_filter(lowcut, highcut), self.sample_rate)
        filtsig.amplitude_env()

        env = filtsig.env
        env_sig = TS(env, self.sample_rate)
        fft_result = np.fft.fft(env_sig.signal)
        freq = np.fft.fftfreq(len(env_sig.signal), d=1 / env_sig.sample_rate)

        modified_fft = np.abs(fft_result) ** 1.57 * np.exp(1j * np.angle(fft_result))

        env_sig.signal = np.real(np.fft.ifft(modified_fft))
        env_sig.signal = mean_filter(env_sig.signal, neighbors)

        return env_sig

    def resample_compact(self):

        assignment = self.details[0][3][-1]

        if assignment == 'g':
            max_freq = 10000
            var_normalizer = 500
            if self.crest() > 1.5:
                var_normalizer = 250
        else:
            max_freq = 1500
            var_normalizer = 1000
            if self.crest() > 3.5:
                var_normalizer = 500

        idx_max_freq = max_freq // self.fq[1]
        if self.fq[-1] < max_freq:
            max_freq = int(self.fq[-1])

        try:
            for idx in range(idx_max_freq, 0, -10):
                pad_var = np.var(self.ft[idx - 10:idx] * var_normalizer)
                pad_var = int(pad_var)
                if (pad_var > 1):
                    # nyquist freq: int(self.fq[idx])*2
                    new_sample_rate = int(self.fq[idx]) * 2
                    break

            resampled_signal = self.resample(self.sample_rate, new_sample_rate)
        except:
            resampled_signal = self.signal.tolist()
            new_sample_rate = self.sample_rate

        resampled_signal = np.round(resampled_signal,3)

        self.resampled_signal = resampled_signal.tolist()
        self.resampled_signal_sr = new_sample_rate

    def bandpass_filter(self, lowcut, highcut, order=5):

        from scipy.signal import butter, filtfilt

        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist

        try:
            b, a = butter(order, [low, high], btype='band')
        except:
            b, a = butter(order, [low, int(self.fq[-50]) / nyquist], btype='band')

        filtered_data = filtfilt(b, a, self.signal)

        return filtered_data

    import numpy as np

    def spectral_energy(self, freq_range=None):
        """
        Computes spectral energy of a signal.
        
        Parameters:
            signal (np.ndarray): Time-domain signal (1D).
            sampling_rate (float): Sampling rate in Hz.
            freq_range (tuple): Optional (min_freq, max_freq) in Hz to compute energy within a range.
            
        Returns:
            float: Spectral energy.
        """
        power_spectrum = self.ft ** 2
        freqs = self.fq

        if freq_range:
            min_f, max_f = freq_range
            freq_mask = (freqs >= min_f) & (freqs <= max_f)
            spectral_energy = np.sum(power_spectrum[freq_mask])
        else:
            spectral_energy = np.sum(power_spectrum)

        return spectral_energy

    def export(self, path: str) -> None:
        """
        Saves the signal to a text file at the given path.

        The first 8 lines of the file contain the details of the signal in the same
        format as the original text file.

        The rest of the file contains the signal data, one value per line.

        :param path: The path to save the file at.
        :type path: str
        """
        import os
        try:
            dir_path = os.path.dirname(path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(path, 'w') as f:
                # Write DataFrame rows
                for row in self.details.itertuples(index=False):
                    f.write(' '.join(map(str, row)) + '\n')

                # Write list values
                for item in self.signal:
                    f.write(str(item) + '\n')
            print(f'Successfully exported signal to {path}')

        except Exception as e:
            print(f'Error saving file as {path}: {e}')

    def describe(self):
        print('__________________________________')
        print('|          O V E R A L L         |')
        print('|________________________________|')

        print(f'| RMS:            |     {self.rms():0.3f}    |')
        print('|_________________|______________|')

        print(f'| Crest:          |     {self.crest():0.3f}    |')
        print('|_________________|______________|')

        print(f'| Kurt:           |     {self.kurt():0.3f}   |')
        print('|_________________|______________|')

        print(f'| Peak:           |     {self.rms():0.3f}    |')
        print('|_________________|______________|')

        print(f'| Peak to Peak:   |     {self.peak_to_peak():0.3f}   |')
        print('|_________________|______________|')
        print(f'| Length (sec):   |     {self.get_length():0.3f}    |')
        print('|_________________|______________|')

        print('')
        print('')

        print('__________________________________')
        print('|      T I M E - S I G N A L     |')
        print('|________________________________|')
        self.sig_plt()

        print('')
        print('')

        print('__________________________________')
        print('|              F F T             |')
        print('|________________________________|')
        self.fftransform()
        self.get_1x_freq()
        self.get_xs_freqs(3)
        self.fft_plt()

        print('__________________________________')
        print(f'| 1X: {self.x_args[0]:0.2f} Hz | Value: {self.x_vals[0]:0.2f} amp |')
        print('|______________|_________________|')
        print(f'| 2X: {self.x_args[1]:0.2f} Hz | Value: {self.x_vals[1]:0.2f} amp |')
        print('|______________|_________________|')
        print(f'| 3X: {self.x_args[2]:0.2f} Hz | Value: {self.x_vals[2]:0.2f} amp |')
        print('|______________|_________________|')

        print('')
        print('')

        print('__________________________________')
        print('|            S T F T             |')
        print('|________________________________|')
        self.stftransform()
        self.stft_plt()

        print('')
        print('')

        print('__________________________________')
        print('|         C I R C U L A R        |')
        print('|________________________________|')
        self.circular_plt(5)
