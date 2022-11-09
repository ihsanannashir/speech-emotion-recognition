import os
import time
import numpy as np
import pandas
import matplotlib.pyplot as plt
import speechpy
import pywt
import scipy
import scipy.fftpack as fft
from scipy.io import wavfile
from scipy.signal import get_window
import codecs, json

def normalize_audio(audio):
    audio = audio/np.max(np.abs(audio))
    return audio

def frame_audio(audio, FFT_size = 2048, hop_size = 10, sample_rate=44100):
    audio = np.pad(audio, int(FFT_size/2), mode='reflect')
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len)+ 1
    frames = np.zeros((frame_num, FFT_size))
    for n in range(frame_num):
        frames[n] = audio[n*frame_len : n*frame_len+FFT_size]
        return frames

def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def met_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = met_to_freq(mels)
    return np.floor((FFT_size) / sample_rate * freqs).astype(int), freqs

def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points)-2, int((FFT_size/2))))
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
    filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    return filters

def replaceZeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)
    for i in range(1, dct_filter_num): 
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
    return basis

data = {
    "mapping": [],
    "labels": [],
    "wfcc": []
}
kelas = ['HI', 'LO', 'MD']
print("--Mulai--")
lokasiDataBaru = "anger/"

count = 0
datacount = 91*3
fiturmean = np.empty((40+1, datacount))
x = 0

for i in os.listdir(lokasiDataBaru):
    for f in os.listdir(lokasiDataBaru+'/'+i):
        print("\nEkstraksi fitur suara dengan WFCC")
        time.sleep(0.1)
        print("Memproses :",f)
        sample_rate, audio = wavfile.read(lokasiDataBaru + i + '/' + f)
        waktuSekarang = time.time()
        print("\t - Membaca audio...\t\t\t\t(done)")

        if (len(audio.shape) > 1):
            audio1 = normalize_audio(audio[:,0])
        else:
            audio1 = normalize_audio(audio)

        audiohasil2 = audio1
        hop_size = 12
        FFT_size = 2048
        audio_framed = frame_audio(audiohasil2, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
        print("\t - Audio Framing...\t\t\t\t(done)")
        
        window = get_window("hamming", FFT_size, fftbins=True)
        audio_win = audio_framed * window
        print("\t - Windowing...\t\t\t\t\t(done)")
        
        #--FFT--
        # audio_winT = np.transpose(audio_win)
        # audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
        # for n in range(audio_fft.shape[1]):
        #     audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
        # audio_fft = np.transpose(audio_fft)
        # print("\t - Fast Fourier Transform...\t\t\t(done)")

        #--Wavelet Transform--
        audio_winT = np.transpose(audio_win)
        coeffs = pywt.wavedec(audio_winT, 'bior6.8', mode='sym', level=2);  # DWT
        cA, cD1, cD2 = coeffs
        audio_wavelet = pywt.waverec(coeffs, 'bior6.8', mode='sym')
        audio_wavelet = np.transpose(audio_wavelet)
        print("\t - Wavelet Transform...\t\t\t\t(done)")
            
        audio_power = np.square(np.abs(audio_wavelet))
        print("\t - Menghitung Audio Power...\t\t\t(done)")
            
        freq_min = 0
        freq_high = sample_rate / 2
        mel_filter_num = 10

        filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, 4096, sample_rate)
        filters = get_filters(filter_points, 4096)
        enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
        filters *= enorm[:, np.newaxis]
        print("\t - Menghitung Filter Point...\t\t\t(done)")
            
        audio_filtered = np.dot(filters, np.transpose(audio_power))
        prob = replaceZeroes(audio_filtered)
        audio_log = 10.0 * np.log10(audio_filtered)
        print("\t - Melakukan Filterisasi Sinyal...\t\t(done)")
            
        dct_filter_num = 40
        dct_filters = dct(dct_filter_num, mel_filter_num)
        cepstral_coefficents = np.dot(dct_filters, audio_log)
        print("\t - Generate Nilai Cepstral Coefficient...\t(done)")

        cepstral_coefficents = speechpy.processing.cmvn(cepstral_coefficents,True)
            
        for xpos in range(len(cepstral_coefficents)):
            sigmax = 0
            for xn in cepstral_coefficents[xpos,:]:
                sigmax += xn
            fiturmean[xpos,count] = sigmax/len(np.transpose(cepstral_coefficents))
            fiturmean[-1,count] = x
        count+=1
        print("\t - Normalisasi Nilai Cepstral Coefficient...\t(done)")

        #--BUAT JADI XLSX--
        indextable = []
        for a in range(40):
            indextable.append("fitur" + str(a+1))
        indextable.append("klasifikasi")

        df = pandas.DataFrame(np.transpose(fiturmean),columns=indextable)
        for y in range(len(fiturmean[40])):
            if y < 91:
                df.loc[[y], "klasifikasi"] = 'HIGH'
            elif y >= 91 and y < 183: 
                df.loc[[y],"klasifikasi"] = 'LOW'
            elif y>=183: 
                df.loc[[y],"klasifikasi"] = 'MIDDLE'

        df.to_excel("WFCCTrain.xlsx", index=False)
        print("Proses ",f, " selesai")

        #--BUAT JADI JSON (HARUS BUAT DATA MENTAH)--
        # df = pandas.DataFrame(np.transpose(fiturmean))
        # for y in range(len(df)):
        #     jsonlist = df.values.tolist()
        #     file_path = "WFCCTrain.json"
        #     data["wfcc"].append(jsonlist)
        #     if i == kelas[0]:
        #         data["labels"].append('0')
        #     elif i == kelas[1]:
        #         data["labels"].append('1')
        #     else:
        #         data["labels"].append('2')
        #     json.dump(data, codecs.open(file_path, 'w', encoding='utf-8'),
        #         separators=(',', ':'),
        #         sort_keys=True,
        #         indent=4)
        # print("Proses ",f, " selesai")

print("--Selesai--")