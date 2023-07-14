import json 
import os 
import librosa 
import tensorflow as tf 
import numpy as np 

SAVED_MODEL_PATH='model.h5'
SAMPLES_TO_CONSIDER=22050
file_path='dog_sound2.wav'

class sound_classification_service:
    model=tf.keras.models.load_model('model.h5')
    _mappings=["air_conditioner","car_horn","children_playing","dog_bark","drilling","engine_idling","gun_shot",
               "jack_hammer","siren","street_music"]
    _instance=None

    def preprocess(self,file_path,n_mfcc=128,n_fft=2048,hop_length=512):
        signal,sr=librosa.load(file_path)
        print(len(signal))
        #if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
        signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
        Mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
            
        return Mfccs.T
    
    def predict(self,file_path):
        mffcs=self.preprocess(file_path)
        MFCCs=mffcs[np.newaxis,...,np.newaxis]
        predictions=self.model.predict(MFCCs)
        predicted_index=np.argmax(predictions)
        predicted_keyword=self._mappings[predicted_index]
        return predicted_keyword

'''
def Sound_Classification_Service():

    if _sound_classification_service is None:
        _sound_classification_service._instance=_sound_classification_service()
        _sound_classification_service.model=tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _sound_classification_service._instance
'''

if __name__=="__main__":
    scs=sound_classification_service()
    keyword=scs.predict('dog_sound2.wav')
    print("keyword:",keyword)
