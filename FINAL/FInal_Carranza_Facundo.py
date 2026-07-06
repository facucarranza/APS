import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal import butter, filtfilt
from scipy.signal import freqz
from scipy.stats import kurtosis, skew


def bandpass_filter(signal, Fs, lowcut=20, highcut=80, order=2):
    nyquist = Fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


def plot_phase(t, H, B1, B2, B3, title):
    plt.figure(figsize=(12,5))
    plt.plot(t, H, label="Healthy")
    plt.plot(t, B1, label="1 BRB")
    plt.plot(t, B2, label="2 BRB")
    plt.plot(t, B3, label="3 BRB")
    plt.title(title)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Corriente")
    #plt.xlim(0, 0.3)
    plt.grid()
    plt.legend()
    plt.show()

# ==========================
# FFT
# ==========================

def compute_fft(signal, Fs):
    N = len(signal)
    # quitar media
    signal = signal - np.mean(signal)
    # ventana de Hanning
    window = np.hanning(N)
    signal = signal * window
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/Fs)
    idx = freqs > 0
    return freqs[idx], np.abs(fft[idx]) / N

# ==========================
# WELCH
# ==========================

def compute_psd(signal, Fs):
    signal = signal - np.mean(signal)
    f, Pxx = welch(
        signal,
        fs=Fs,
        window='hann',
        nperseg=2048,
        noverlap=512
    )
    return f, Pxx


def analyze_phase(H, B1, B2, B3, Fs, title):

    # ======================
    # FFT
    # ======================
    f_H, X_H = compute_fft(H, Fs)
    f_1B, X_1B = compute_fft(B1, Fs)
    f_2B, X_2B = compute_fft(B2, Fs)
    f_3B, X_3B = compute_fft(B3, Fs)

    plt.figure(figsize=(12,5))
    plt.semilogy(f_H, X_H, label="Healthy")
    plt.semilogy(f_1B, X_1B, label="1 BRB")
    plt.semilogy(f_2B, X_2B, label="2 BRB")
    plt.semilogy(f_3B, X_3B, label="3 BRB")
    plt.xlim(0, 400)
    plt.title(f"FFT - {title}")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud")
    plt.grid()
    plt.legend()
    plt.show()

    # ======================
    # WELCH
    # ======================
    f_H, P_H = compute_psd(H, Fs)
    f_1B, P_1B = compute_psd(B1, Fs)
    f_2B, P_2B = compute_psd(B2, Fs)
    f_3B, P_3B = compute_psd(B3, Fs)

    plt.figure(figsize=(12,5))
    plt.semilogy(f_H, P_H, label="Healthy")
    plt.semilogy(f_1B, P_1B, label="1 BRB")
    plt.semilogy(f_2B, P_2B, label="2 BRB")
    plt.semilogy(f_3B, P_3B, label="3 BRB")
    plt.xlim(0, 400)
    plt.title(f"Welch PSD - {title}")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Potencia")
    plt.grid()
    plt.legend()
    plt.show()

# ==========================
# ARCHIVO EXCEL
# ==========================

archivo = "M24Ns_29Nb.xlsx"

# ==========================
# LEER SOLO LAS HOJAS NECESARIAS
# ==========================

df_H = pd.read_excel(archivo, sheet_name="H", header=None)
df_1B = pd.read_excel(archivo, sheet_name="1B", header=None)
df_2B = pd.read_excel(archivo, sheet_name="2B", header=None)
df_3B = pd.read_excel(archivo, sheet_name="3B", header=None)

# ==========================
# FRECUENCIA DE MUESTREO
# ==========================

Fs = 2000

# ==========================
# EXTRAER FASES
# ==========================

# FASE A
A_H = df_H.iloc[:,1].values
A_1B = df_1B.iloc[:,1].values
A_2B = df_2B.iloc[:,1].values
A_3B = df_3B.iloc[:,1].values

# FASE B
B_H = df_H.iloc[:,2].values
B_1B = df_1B.iloc[:,2].values
B_2B = df_2B.iloc[:,2].values
B_3B = df_3B.iloc[:,2].values

# FASE C
C_H = df_H.iloc[:,3].values
C_1B = df_1B.iloc[:,3].values
C_2B = df_2B.iloc[:,3].values
C_3B = df_3B.iloc[:,3].values

# ==========================
# ELIMINAR TRANSITORIO
# ==========================

start = int(0.2 * Fs)

# FASE A
A_H = A_H[start:]
A_1B = A_1B[start:]
A_2B = A_2B[start:]
A_3B = A_3B[start:]

# FASE B
B_H = B_H[start:]
B_1B = B_1B[start:]
B_2B = B_2B[start:]
B_3B = B_3B[start:]

# FASE C
C_H = C_H[start:]
C_1B = C_1B[start:]
C_2B = C_2B[start:]
C_3B = C_3B[start:]

# ==========================
# VECTOR DE TIEMPO
# ==========================

t = np.arange(len(A_H)) / Fs

# ==========================
# GRAFICAR
# ==========================
"""
plot_phase(t, A_H, A_1B, A_2B, A_3B, "Fase A")
plot_phase(t, B_H, B_1B, B_2B, B_3B, "Fase B")
plot_phase(t, C_H, C_1B, C_2B, C_3B, "Fase C")
"""
"""
analyze_phase(A_H, A_1B, A_2B, A_3B, Fs,"Fase A")
analyze_phase(B_H, B_1B, B_2B, B_3B, Fs, "Fase B")
analyze_phase(C_H, C_1B, C_2B, C_3B, Fs, "Fase C")
"""

b, a = butter(4, [20/(Fs/2), 80/(Fs/2)], btype='band')
w, h = freqz(b, a, worN=4096)

plt.figure(figsize=(10,4))
plt.plot(w*Fs/(2*np.pi), 20*np.log10(np.abs(h) + 1e-300))
plt.grid()
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Ganancia [dB]")
plt.title("Respuesta en frecuencia del filtro pasabanda")
plt.xlim(0,200)
plt.ylim(-400,0)
plt.show()

# ==========================
# FILTRO PASABANDA 20-80 Hz
# ==========================

# FASE A
A_H_filt = bandpass_filter(A_H, Fs)
A_1B_filt = bandpass_filter(A_1B, Fs)
A_2B_filt = bandpass_filter(A_2B, Fs)
A_3B_filt = bandpass_filter(A_3B, Fs)

# FASE B
B_H_filt = bandpass_filter(B_H, Fs)
B_1B_filt = bandpass_filter(B_1B, Fs)
B_2B_filt = bandpass_filter(B_2B, Fs)
B_3B_filt = bandpass_filter(B_3B, Fs)

# FASE C
C_H_filt = bandpass_filter(C_H, Fs)
C_1B_filt = bandpass_filter(C_1B, Fs)
C_2B_filt = bandpass_filter(C_2B, Fs)
C_3B_filt = bandpass_filter(C_3B, Fs)

analyze_phase(
    A_H_filt, A_1B_filt, A_2B_filt, A_3B_filt,
    Fs, "Fase A - Pasabanda 20-80 Hz"
)

analyze_phase(
    B_H_filt, B_1B_filt, B_2B_filt, B_3B_filt,
    Fs, "Fase B - Pasabanda 20-80 Hz"
)

analyze_phase(
    C_H_filt, C_1B_filt, C_2B_filt, C_3B_filt,
    Fs, "Fase C - Pasabanda 20-80 Hz"
)

# ==========================
# EXTRACCION DE CARACTERISTICAS
# ==========================

def band_energy(f_arr, Pxx_arr, f_low, f_high):
    # energia (integral de Pxx) entre f_low y f_high
    mask = (f_arr >= f_low) & (f_arr <= f_high)
    if mask.sum() < 2:
        return 0.0
    return np.trapezoid(Pxx_arr[mask], f_arr[mask])


def fft_peak(f_arr, X_arr, f_low, f_high):
    # amplitud maxima de la FFT dentro de una banda
    mask = (f_arr >= f_low) & (f_arr <= f_high)
    if mask.sum() == 0:
        return 0.0
    return X_arr[mask].max()


def extract_features(signal, Fs):

    f, Pxx = compute_psd(signal, Fs)

    peak_idx = np.argmax(Pxx)
    peak_freq = f[peak_idx]
    peak_power = Pxx[peak_idx]

    band_power = np.trapezoid(Pxx, f)
    centroid = np.sum(f * Pxx) / np.sum(Pxx)
    bandwidth = np.sqrt(np.sum(((f - centroid)**2) * Pxx) / np.sum(Pxx))

    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms
    std = np.std(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)

    # energia del fundamental (45-55 Hz) y de las sidebands de BRB
    # (40-45 Hz y 55-60 Hz, a ambos lados del fundamental)
    E_fund = band_energy(f, Pxx, 45, 55)
    E_sb_left = band_energy(f, Pxx, 40, 45)
    E_sb_right = band_energy(f, Pxx, 55, 60)
    E_sb = E_sb_left + E_sb_right

    # indice de sidebands en dB: crece con la severidad de la falla
    sideband_index = 10 * np.log10((E_sb + 1e-30) / (E_fund + 1e-30))

    # entropia espectral: que tan concentrado/disperso esta el espectro
    Pxx_norm = Pxx / (Pxx.sum() + 1e-30)
    Pxx_norm = np.where(Pxx_norm > 0, Pxx_norm, 1e-30)
    spectral_entropy = -np.sum(Pxx_norm * np.log2(Pxx_norm))

    # amplitud del fundamental y de las sidebands medida con FFT
    # (mayor resolucion en frecuencia que Welch)
    f_fft, X_fft = compute_fft(signal, Fs)
    fund_amp = fft_peak(f_fft, X_fft, 48, 52)
    sb_left_amp = fft_peak(f_fft, X_fft, 40, 45)
    sb_right_amp = fft_peak(f_fft, X_fft, 55, 60)

    fault_index_dB = 20 * np.log10(
        (sb_left_amp + sb_right_amp + 1e-30) / (2 * fund_amp + 1e-30)
    )

    return {
        "PeakFreq": peak_freq,
        "PeakPower": peak_power,
        "BandPower": band_power,
        "SpectralCentroid": centroid,
        "SpectralBandwidth": bandwidth,
        "RMS": rms,
        "STD": std,
        "Skewness": skewness,
        "Kurtosis": kurt,
        "CrestFactor": crest_factor,
        "Energy_45_55Hz": E_fund,
        "Energy_40_45Hz": E_sb_left,
        "Energy_55_60Hz": E_sb_right,
        "SidebandIndex_dB": sideband_index,
        "SpectralEntropy": spectral_entropy,
        "FaultIndex_FFT_dB": fault_index_dB,
    }


results_A = pd.DataFrame([
    extract_features(A_H_filt, Fs),
    extract_features(A_1B_filt, Fs),
    extract_features(A_2B_filt, Fs),
    extract_features(A_3B_filt, Fs)
], index=["Healthy","1BRB","2BRB","3BRB"])

print("\nFASE A")
print(results_A)

results_B = pd.DataFrame([
    extract_features(B_H_filt, Fs),
    extract_features(B_1B_filt, Fs),
    extract_features(B_2B_filt, Fs),
    extract_features(B_3B_filt, Fs)
], index=["Healthy","1BRB","2BRB","3BRB"])

print("\nFASE B")
print(results_B)

results_C = pd.DataFrame([
    extract_features(C_H_filt, Fs),
    extract_features(C_1B_filt, Fs),
    extract_features(C_2B_filt, Fs),
    extract_features(C_3B_filt, Fs)
], index=["Healthy","1BRB","2BRB","3BRB"])

print("\nFASE C")
print(results_C)

ratio_A = results_A.div(results_A.loc["Healthy"])
print(ratio_A)

ratio_B = results_B.div(results_B.loc["Healthy"])
print(ratio_B)

ratio_C = results_C.div(results_C.loc["Healthy"])
print(ratio_C)

# ==========================
# GRAFICO DE LAS FEATURES NUEVAS
# ==========================

NEW_FEATURES = [
    "Energy_40_45Hz",
    "Energy_45_55Hz",
    "Energy_55_60Hz",
    "SidebandIndex_dB",
    "SpectralEntropy",
    "FaultIndex_FFT_dB",
]

LABELS = ["Healthy", "1BRB", "2BRB", "3BRB"]

# promedio de las tres fases para cada feature nueva
results_mean = (results_A + results_B + results_C) / 3

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for ax, feat in zip(axes, NEW_FEATURES):
    vals = results_mean.loc[LABELS, feat].values
    ax.bar(LABELS, vals)
    ax.set_title(feat)
    ax.grid(axis="y")

plt.tight_layout()
plt.show()

print("\nNUEVAS FEATURES - FASE A")
print(results_A[NEW_FEATURES])

print("\nNUEVAS FEATURES - FASE B")
print(results_B[NEW_FEATURES])

print("\nNUEVAS FEATURES - FASE C")
print(results_C[NEW_FEATURES])

print("\nRATIO VS HEALTHY - PROMEDIO 3 FASES")
ratio_mean = results_mean.div(results_mean.loc["Healthy"])
print(ratio_mean[NEW_FEATURES])