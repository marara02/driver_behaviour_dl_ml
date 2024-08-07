import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score, roc_curve
from scipy.stats import skew, kurtosis
from scipy.fft import fft

from keras.layers import Activation, Dense, Dropout, Embedding, LSTM
from keras import regularizers
import re
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras, lite
from tensorflow.keras.layers import Dense, Dropout, Flatten, ConvLSTM2D, LSTM, RepeatVector, BatchNormalization
import keras_tuner as kt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pylab import rcParams
from matplotlib import rc
from multiprocessing import cpu_count
import pickle

tf.random.set_seed(42)
TIMESTEPS = 10 # the number of sample to be fed to the NN
FEATURES = 6
LABELS = 3
N_RECORDS = 11
BATCH_SIZE = 250   

df_train = pd.read_csv('dataset/train_motion_data.csv')
df_test = pd.read_csv('dataset/test_motion_data.csv')

df_train = df_train.dropna()
df_test = df_test.dropna()

# Merge class 'Slow' to 'Normal' class
df_train_2_cl = df_train.copy()
df_test_2_cl = df_test.copy()

two_class_mapping = {'SLOW': 'NORMAL'}
df_train_2_cl['Class'] = df_train_2_cl['Class'].replace(two_class_mapping)
df_test_2_cl['Class'] = df_test_2_cl['Class'].replace(two_class_mapping)


def extract_features(df, window_size):
    features = []
    labels = []
    for i in range(0, len(df), window_size):
        window = df.iloc[i:i+window_size]
        if len(window) == window_size:
            acc_x = window['AccX'].values
            acc_y = window['AccY'].values
            acc_z = window['AccZ'].values
            gyro_x = window['GyroX'].values
            gyro_y = window['GyroY'].values
            gyro_z = window['GyroZ'].values

            # Time domain features
            feature_vector = [
                np.mean(acc_x), np.std(acc_x), skew(acc_x), kurtosis(acc_x),
                np.mean(acc_y), np.std(acc_y), skew(acc_y), kurtosis(acc_y),
                np.mean(acc_z), np.std(acc_z), skew(acc_z), kurtosis(acc_z),
                np.mean(gyro_x), np.std(gyro_x), skew(gyro_x), kurtosis(gyro_x),
                np.mean(gyro_y), np.std(gyro_y), skew(gyro_y), kurtosis(gyro_y),
                np.mean(gyro_z), np.std(gyro_z), skew(gyro_z), kurtosis(gyro_z)
            ]

            # Frequency domain features (FFT)
            fft_acc_x = np.abs(fft(acc_x))
            fft_acc_y = np.abs(fft(acc_y))
            fft_acc_z = np.abs(fft(acc_z))
            fft_gyro_x = np.abs(fft(gyro_x))
            fft_gyro_y = np.abs(fft(gyro_y))
            fft_gyro_z = np.abs(fft(gyro_z))
            feature_vector.extend([
                np.mean(fft_acc_x), np.std(fft_acc_x),
                np.mean(fft_acc_y), np.std(fft_acc_y),
                np.mean(fft_acc_z), np.std(fft_acc_z),
                np.mean(fft_gyro_x), np.std(fft_gyro_x),
                np.mean(fft_gyro_y), np.std(fft_gyro_y),
                np.mean(fft_gyro_z), np.std(fft_gyro_z)
            ])
            
            features.append(feature_vector)
            labels.append(window['Class'].mode()[0])  # Assuming class labels are consistent within a window

    return np.array(features), np.array(labels)

df_train_2_normal = df_train_2_cl.loc[df_train_2_cl['Class'] == "NORMAL"]
df_train_2_aggressive = df_train_2_cl.loc[df_train_2_cl['Class'] == "AGGRESSIVE"]

df_test_2_normal = df_test_2_cl.loc[df_test_2_cl['Class'] == "NORMAL"]
df_test_2_aggressive = df_test_2_cl.loc[df_test_2_cl['Class'] == "AGGRESSIVE"]

df_train_2_normal = df_train_2_normal.iloc[N_RECORDS:]
df_train_2_normal = df_train_2_normal.iloc[:-N_RECORDS]

df_train_2_aggressive = df_train_2_aggressive.iloc[N_RECORDS:]
df_train_2_aggressive = df_train_2_aggressive.iloc[:-N_RECORDS]

df_test_2_normal = df_test_2_normal.iloc[N_RECORDS:]
df_test_2_normal = df_test_2_normal.iloc[:-N_RECORDS]

df_test_2_aggressive = df_test_2_aggressive.iloc[N_RECORDS:]
df_test_2_aggressive = df_test_2_aggressive.iloc[:-N_RECORDS]

df_train_2_normal = df_train_2_normal.tail(2500)
df_train_2_aggressive = df_train_2_aggressive.tail(1090)

df_test_2_normal = df_test_2_normal.tail(2240)
df_test_2_aggressive = df_test_2_aggressive.tail(790)

# Features
X_train_2_normal = df_train_2_normal.iloc[: , :FEATURES]
X_train_2_aggressive = df_train_2_aggressive.iloc[: , :FEATURES]

X_test_2_normal = df_test_2_normal.iloc[: , :FEATURES]
X_test_2_aggressive = df_test_2_aggressive.iloc[: , :FEATURES]

# Labels
y_train_2_normal = df_train_2_normal.Class
y_train_2_aggressive = df_train_2_aggressive.Class

y_test_2_normal = df_test_2_normal.Class
y_test_2_aggressive = df_test_2_aggressive.Class

X_train_2 = pd.concat([X_train_2_normal, X_train_2_aggressive])
y_train_2 = pd.concat([y_train_2_normal, y_train_2_aggressive])

X_test_2 = pd.concat([X_test_2_normal, X_test_2_aggressive])
y_test_2 = pd.concat([y_test_2_normal, y_test_2_aggressive])

scaler = StandardScaler(with_mean=True, with_std=True)
X_train_2 = scaler.fit_transform(X_train_2)
X_test_2 = scaler.fit_transform(X_test_2)

labelEncoder_2 = LabelEncoder()
y_train_2 = labelEncoder_2.fit_transform(y_train_2)
y_test_2 = labelEncoder_2.transform(y_test_2)

y_train_2 = to_categorical(y_train_2, num_classes=LABELS)
y_test_2 = to_categorical(y_test_2, num_classes=LABELS)

train_samples_two = X_train_2.shape[0] // TIMESTEPS
X_train_two = X_train_2.reshape(train_samples_two, TIMESTEPS, FEATURES)

test_samples_two = X_test_2.shape[0] // TIMESTEPS
X_test_two = X_test_2.reshape(test_samples_two, TIMESTEPS, FEATURES)

y_train_two = y_train_2[::TIMESTEPS]
y_test_two = y_test_2[::TIMESTEPS]


def model_builder(hp):
    model = tf.keras.Sequential()

    # First LSTM layer with input shape and batch normalization
    model.add(LSTM(hp.Int('input_unit', min_value=32, max_value=512, step=32), 
                   input_shape=(TIMESTEPS, FEATURES), return_sequences=True))
    model.add(BatchNormalization())

    # Additional LSTM layers with dropout and batch normalization
    for i in range(hp.Int('n_layers', 1, 4)):
        model.add(LSTM(
            hp.Int(f'lstm_{i}_units', min_value=32, max_value=512, step=32),
            dropout=hp.Float(f'dropout_{i}_rate', min_value=0, max_value=0.5, step=0.1),
            return_sequences=True
        ))
        model.add(BatchNormalization())

    # Final LSTM layer without return_sequences
    model.add(LSTM(hp.Int('lstm_output_neurons', min_value=32, max_value=512, step=32), return_sequences=False))
    model.add(BatchNormalization())

    # Dropout for regularization
    model.add(Dropout(hp.Float('output_dropout_rate', min_value=0, max_value=0.5, step=0.1)))

    # Dense layer
    model.add(Dense(hp.Int('dense_neurons', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(BatchNormalization())

    # Output layer with softmax activation
    model.add(Dense(LABELS, activation='softmax'))

    # Learning rate selection
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])

    # Model compilation with additional metrics
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model


tuner_2 = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=50,
                     factor=3,
                     directory='lstm_dir_2',
                     project_name='driving_behavior')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner_2.search(
        x=X_train_two,
        y=y_train_two,
        epochs=80,
        validation_data=(X_test_two, y_test_two),
        callbacks=[stop_early], 
        shuffle=True
)

best_hps_2 = tuner_2.get_best_hyperparameters(num_trials=1)[0]

model_2 = tuner_2.hypermodel.build(best_hps_2)
history_2 = model_2.fit(X_train_two, y_train_two, epochs=80, validation_data=(X_test_two, y_test_two), 
    callbacks=[stop_early], 
    shuffle=True)

model_2.save("model_drive_style.h5")