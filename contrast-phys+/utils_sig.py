import numpy as np
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt

def butter_bandpass(sig, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter
    
    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    y = filtfilt(b, a, sig)
    return y

def butter_bandpass_batch(sig_list, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter (batch version)
    # signals are in the sig_list

    y_list = []
    
    for sig in sig_list:
        sig = np.reshape(sig, -1)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, sig)
        y_list.append(y)
    return np.array(y_list)

def hr_fft(sig, fs, harmonics_removal=True):
    # get heart rate by FFT
    # return both heart rate and PSD

    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))
    low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
    high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
    sig_f_original = sig_f.copy()
    
    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    f_hr1 = peak_idx1 / sig.shape[0] * fs
    hr1 = f_hr1 * 60

    f_hr2 = peak_idx2 / sig.shape[0] * fs
    hr2 = f_hr2 * 60
    if harmonics_removal:
        if np.abs(hr1-2*hr2)<10:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1

    x_hr = np.arange(len(sig))/len(sig)*fs*60
    return hr, sig_f_original, x_hr

def hr_fft_batch(sig_list, fs, harmonics_removal=True):
    # get heart rate by FFT (batch version)
    # return both heart rate and PSD

    hr_list = []
    for sig in sig_list:
        sig = sig.reshape(-1)
        sig = sig * signal.windows.hann(sig.shape[0])
        sig_f = np.abs(fft(sig))
        low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
        high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
        sig_f_original = sig_f.copy()
        
        sig_f[:low_idx] = 0
        sig_f[high_idx:] = 0

        peak_idx, _ = signal.find_peaks(sig_f)
        sort_idx = np.argsort(sig_f[peak_idx])
        sort_idx = sort_idx[::-1]

        peak_idx1 = peak_idx[sort_idx[0]]
        peak_idx2 = peak_idx[sort_idx[1]]

        f_hr1 = peak_idx1 / sig.shape[0] * fs
        hr1 = f_hr1 * 60

        f_hr2 = peak_idx2 / sig.shape[0] * fs
        hr2 = f_hr2 * 60
        if harmonics_removal:
            if np.abs(hr1-2*hr2)<10:
                hr = hr2
            else:
                hr = hr1
        else:
            hr = hr1

        # x_hr = np.arange(len(sig))/len(sig)*fs*60
        hr_list.append(hr)
    return np.array(hr_list)

def normalize(x):
    return (x-x.mean())/x.std()

def SNR_get(waveform, gt_hr, fs, filtered=False):
    waveform = np.reshape(waveform, -1)
    if filtered:
        waveform = butter_bandpass(waveform, 0.6, 4, 60)
    N = waveform.shape[0]
    
#     low_idx = np.round(0.6 / fs * waveform.shape[0]).astype('int')
#     high_idx = np.round(4 / fs * waveform.shape[0]).astype('int')
    
#     waveform[:low_idx] = 0
#     waveform[high_idx:] = 0
    
    bin1 = round(5/60/fs*N)
    bin2 = round(10/60/fs*N)

    f1 = gt_hr / 60
    f2 = f1 * 2

    bc1 = round(f1*N/fs)
    bc2 = round(f2*N/fs)

    window = signal.windows.hann(N)
    win_waveform = waveform * window
    waveform_f = np.abs(fft(win_waveform))**2

    total_power = np.sum(waveform_f)
    signal_power1 = 2*np.sum(waveform_f[bc1-bin1:bc1+bin1])
    signal_power2 = 2*np.sum(waveform_f[bc2-bin2:bc2+bin2])

    signal_power = signal_power1 + signal_power2
    noise_power = total_power - signal_power

    snr = 10*np.log10(signal_power/noise_power)
    return snr

def hr_fft_zp(sig, fs, harmonics_removal=True, nfft_factor=8):
    """
    High-resolution HR estimation using zero-padding FFT
    Zero-padding increases frequency resolution without requiring longer signals
    
    Parameters:
        sig: input signal
        fs: sampling frequency
        harmonics_removal: whether to remove harmonics
        nfft_factor: zero-padding factor (nfft = nfft_factor * N)
    
    Returns:
        hr: heart rate in BPM
        sig_f_original: original FFT spectrum
        x_hr: frequency axis in BPM
    """
    sig = sig.reshape(-1)
    N = sig.shape[0]
    
    # Apply window
    sig_windowed = sig * signal.windows.hann(N)
    
    # Zero-padding: pad to nfft_factor * N
    nfft = nfft_factor * N
    sig_padded = np.zeros(nfft)
    sig_padded[:N] = sig_windowed
    
    # FFT
    sig_f = np.abs(fft(sig_padded))
    sig_f_original = sig_f.copy()
    
    # Frequency mask: [0.6, 4] Hz
    low_idx = np.round(0.6 / fs * nfft).astype('int')
    high_idx = np.round(4 / fs * nfft).astype('int')
    
    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0
    
    # Find peaks
    peak_idx, _ = signal.find_peaks(sig_f)
    if len(peak_idx) < 2:
        # Fallback: use original method if not enough peaks
        return hr_fft(sig, fs, harmonics_removal)
    
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]
    
    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]
    
    # Convert to frequency and HR
    f_hr1 = peak_idx1 / nfft * fs
    hr1 = f_hr1 * 60
    
    f_hr2 = peak_idx2 / nfft * fs
    hr2 = f_hr2 * 60
    
    if harmonics_removal:
        if np.abs(hr1 - 2*hr2) < 10:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1
    
    x_hr = np.arange(nfft) / nfft * fs * 60
    return hr, sig_f_original, x_hr


def hr_fft_parabolic(sig, fs, harmonics_removal=True):
    """
    High-resolution HR estimation using parabolic interpolation
    Interpolates around the peak to get sub-bin frequency resolution
    
    Parameters:
        sig: input signal
        fs: sampling frequency
        harmonics_removal: whether to remove harmonics
    
    Returns:
        hr: heart rate in BPM
        sig_f_original: original FFT spectrum
        x_hr: frequency axis in BPM
    """
    sig = sig.reshape(-1)
    N = sig.shape[0]
    
    # Apply window
    sig_windowed = sig * signal.windows.hann(N)
    
    # FFT
    sig_f = np.abs(fft(sig_windowed))
    sig_f_original = sig_f.copy()
    
    # Frequency mask: [0.6, 4] Hz
    low_idx = np.round(0.6 / fs * N).astype('int')
    high_idx = np.round(4 / fs * N).astype('int')
    
    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0
    
    # Find peaks
    peak_idx, _ = signal.find_peaks(sig_f)
    if len(peak_idx) < 2:
        # Fallback: use original method if not enough peaks
        return hr_fft(sig, fs, harmonics_removal)
    
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]
    
    peak_idx1_raw = peak_idx[sort_idx[0]]
    peak_idx2_raw = peak_idx[sort_idx[1]]
    
    # Parabolic interpolation for peak 1
    def parabolic_interp(peak_idx, spectrum):
        """Parabolic interpolation around peak"""
        if peak_idx <= 0 or peak_idx >= len(spectrum) - 1:
            return peak_idx
        
        y1 = spectrum[peak_idx - 1]
        y2 = spectrum[peak_idx]
        y3 = spectrum[peak_idx + 1]
        
        # Parabolic fit: y = ax^2 + bx + c
        # Peak location: x = -b/(2a)
        # Using three points: (peak_idx-1, y1), (peak_idx, y2), (peak_idx+1, y3)
        a = (y1 + y3 - 2*y2) / 2
        if abs(a) < 1e-10:
            return peak_idx
        
        b = (y3 - y1) / 2
        peak_offset = -b / (2 * a)
        peak_idx_refined = peak_idx + peak_offset
        
        return peak_idx_refined
    
    peak_idx1 = parabolic_interp(peak_idx1_raw, sig_f)
    peak_idx2 = parabolic_interp(peak_idx2_raw, sig_f)
    
    # Convert to frequency and HR
    f_hr1 = peak_idx1 / N * fs
    hr1 = f_hr1 * 60
    
    f_hr2 = peak_idx2 / N * fs
    hr2 = f_hr2 * 60
    
    if harmonics_removal:
        if np.abs(hr1 - 2*hr2) < 10:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1
    
    x_hr = np.arange(N) / N * fs * 60
    return hr, sig_f_original, x_hr


def compute_fft_peaks(sig, fs):
    """
    计算频谱峰值，并对峰值做抛物线插值以提高频率分辨率
    返回 peaks 列表: [{hr_bpm, power, idx, idx_refined}]
    """
    sig = sig.reshape(-1)
    N = sig.shape[0]
    sig_windowed = sig * signal.windows.hann(N)
    sig_f = np.abs(fft(sig_windowed))

    low_idx = int(np.round(0.6 / fs * N))
    high_idx = int(np.round(4.0 / fs * N))
    sig_f_masked = sig_f.copy()
    sig_f_masked[:low_idx] = 0
    sig_f_masked[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f_masked)
    if len(peak_idx) == 0:
        return []

    peaks = []
    for idx in peak_idx:
        power = sig_f_masked[idx]
        idx_refined = float(idx)
        if 1 <= idx < (N - 1):
            y0 = sig_f_masked[idx - 1]
            y1 = sig_f_masked[idx]
            y2 = sig_f_masked[idx + 1]
            denom = (y0 - 2 * y1 + y2)
            if denom != 0:
                delta = 0.5 * (y0 - y2) / denom
                idx_refined = idx + delta
        hr_bpm = (idx_refined / N) * fs * 60
        peaks.append({
            "hr_bpm": hr_bpm,
            "power": power,
            "idx": idx,
            "idx_refined": idx_refined
        })

    peaks.sort(key=lambda p: p["power"], reverse=True)
    return peaks


def select_hr_from_peaks(peaks, fs, n_samples, use_harmonics_removal=True):
    """
    基于频率分辨率与谐波关系的峰值选择（非硬编码）
    """
    if not peaks:
        return None, {"reason": "no_peaks"}

    resolution_bpm = fs / n_samples * 60
    tol_bpm = 1.5 * resolution_bpm

    for p in peaks:
        p["harm_penalty"] = 0.0

    if use_harmonics_removal:
        for hi in peaks:
            for lo in peaks:
                if lo["hr_bpm"] <= 0:
                    continue
                ratio = hi["hr_bpm"] / lo["hr_bpm"]
                if abs(ratio - 2.0) < (tol_bpm / max(lo["hr_bpm"], 1e-6)) or \
                   abs(ratio - 3.0) < (tol_bpm / max(lo["hr_bpm"], 1e-6)):
                    hi["harm_penalty"] += lo["power"] / (hi["power"] + 1e-9)

    for p in peaks:
        p["score"] = p["power"] / (1.0 + p["harm_penalty"])

    peaks_sorted = sorted(peaks, key=lambda p: p["score"], reverse=True)
    best = peaks_sorted[0]
    return best["hr_bpm"], {
        "reason": "peak_score",
        "resolution_bpm": resolution_bpm,
        "selected_hr": best["hr_bpm"],
        "selected_power": best["power"],
        "selected_score": best["score"]
    }


def es(series, alpha):
    """given a series and alpha, return series of expoentially smoothed points"""
    results = np.zeros_like(series)

    # first value remains the same as series,
    # as there is no history to learn from
    results[0] = series[0] 
    for t in range(1, series.shape[0]):
        results[t] = alpha * series[t] + (1 - alpha) * results[t - 1]

    return results