# Import library yang dibutuhkan
import librosa
import pywt

# Definisikan fungsi wavelet transform
def wavelet_transform(signal, wavelet):
    coeffs = pywt.wavedec(signal, wavelet)
    return coeffs[0]

# Baca sinyal audio
signal, sr = librosa.load("audio.wav")

# Ekstrak fitur MFCC menggunakan FFT
mfcc_fft = librosa.feature.mfcc(signal, sr=sr, n_mfcc=13)

# Ekstrak fitur MFCC menggunakan wavelet transform
wavelet = "db4"
mfcc_wavelet = librosa.feature.mfcc(signal, sr=sr, n_mfcc=13, fft_type=wavelet_transform, fft_type_kwargs={"wavelet": wavelet})

# Tampilkan hasil ekstraksi fitur MFCC menggunakan FFT dan wavelet transform
print("MFCC with FFT:", mfcc_fft)
print("MFCC with wavelet transform:", mfcc_wavelet)
