#! Myenv/bin/python

from hashlib import new
import sys
import  threading
#import bluetooth
import time
import pyaudio
import math
import struct
import wave
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import  StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import warnings
from keras.models import load_model
import tensorflow as tf

from BFCCAldi import convertBFCC
from wfcc import convertWFCC

#BLUETOOTH
addr = 'D0:9C:7A:18:A1:6D'
name = 'Bluetooth_Chat'
uuid = "7D31344C-E08E-463D-B2F3-9FB84DC5B31B"

#EMOTION
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
Threshold = 2

SHORT_NORMALIZE = (1.0/32768.0)
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
swidth = 2
Max_Seconds = 1
TimeoutSignal = ((RATE / chunk * Max_Seconds)+2)
silence = True
FileNameTmp = 'Out.wav'
Time=0
ac = []



# scan_devices = bluetooth.discover_devices()
# print(scan_devices)
service_matches = bluetooth.find_service(name=name, uuid=uuid)
print(service_matches)

# search for the SampleServer service




while len(service_matches) == 0:
    print('searching')
    service_matches = bluetooth.find_service(name=name, uuid=uuid)

first_match = service_matches[0]
print(first_match)
port = first_match["port"]
name = first_match["name"]
host = first_match["host"]

print("Connecting to \"{}\" on {}".format(name, host))

# Create the client socket
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect((host, port))



def receiveData():
    global dataFromApp
    while True:
        dataFromApp = sock.recv(1024).decode()
        print("DATA FROM APP=============  ", dataFromApp)

        if dataFromApp == '3':
            print("turning off")
            break
            sys.exit()
        

# def mainLoop():
#     global dataFromApp
#     i = 0
#     while True:
#         while dataFromApp == '1':
#             # print("JALAN LOOP")
#             # for i in range(10):
#             if dataFromApp == '2':
#                 break
#             else:
#                 sock.send(str(i))
#                 print(i)
#                 time.sleep(1)
#                 i += 1

def StartStream():
    p = pyaudio.PyAudio()
    print("turning on mic")

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    return stream,p

def StopStream(stream,p):
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("turning off mic")

def GetStream(chunk, stream):
    return stream.read(chunk)

def rms(frame):
    count = len(frame)/swidth
    format = "%dh"%(count)
    # short is 16 bit int
    shorts = struct.unpack( format, frame )

    sum_squares = 0.0
    for sample in shorts:
        n = sample * SHORT_NORMALIZE
        sum_squares += n*n
    # compute the rms
    rms = math.pow(sum_squares/count,0.5)
    return rms * 1000





def KeepRecord():
    global dataFromApp
    
    if dataFromApp == '1':
        start = time.time()
        convertBFCC()
        end = time.time()
        print ("Waktu ekstraksi = ", end-start)
        predictBFCC()

    if dataFromApp == '2':
        start = time.time()
        convertWFCC()
        end = time.time()
        print ("Waktu ekstraksi = ", end-start)
        predictWFCC()

def predictBFCC():

    start = time.time()
    #Klasifikasi KNN
    path1 = "BFCCTrain.xlsx"
    path2 = "BFCCTest.xlsx"

    dataset1 = pd.read_excel(path1, header=None)
    dataset2 = pd.read_excel(path2, header=None)
    
    x_train = dataset1.iloc[1:, :40].values
    y_train = dataset1.iloc[1:, 40].values

    x_test = dataset2.iloc[1:, :40].values
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")
    knn.fit(x_train, y_train)
    klasifikasiDataMentah = knn.predict(x_test)
    hasil = np.array_str(klasifikasiDataMentah)

    print (f"Entensitas emosi sedih anda {hasil}")
    send = f"Intensitas emosi sedih anda {hasil}"
    sock.send(send)
    end = time.time()
    print ("Waktu prediksi = ", end-start)


def predictWFCC():
    start = time.time()
    path1 = 'WFCCTrain.xlsx' #data training
    path2 = 'WFCCTest.xlsx' #data test mentah

    dataset1 = pd.read_excel(path1, header=None)
    dataset2 = pd.read_excel(path2, header=None)

    x_train = dataset1.iloc[1:, :40].values
    y_train = dataset1.iloc[1:, 40].values

    x_test = dataset2.iloc[1:, :40].values
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")
    knn.fit(x_train,y_train)
    klasifikasiDataMentah = knn.predict(x_test)

    hasil = np.array_str(klasifikasiDataMentah)
    print (f"Intensitas emosi Marah anda {hasil}")
    send = f"Intensitas emosi Marah anda {hasil}"
    sock.send(send)
    end = time.time()
    print ("Waktu prediksi = ", end-start)

if __name__ == '__main__':
    dataFromApp = ''
    # dataApp = sock.recv(1024).decode()
    # stream, p = StartStream()
    t1 = threading.Thread(target=KeepRecord, args=())
    t2 = threading.Thread(target=receiveData, args=())
    # if dataApp == '1':

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    
