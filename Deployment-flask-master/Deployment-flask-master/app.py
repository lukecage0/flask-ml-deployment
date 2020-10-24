import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import tensorflow as tf
import pickle
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text=[str(x) for x in request.form.values()]
    tweets = pd.read_csv('C:/Users/DARKMASTER/Desktop/Deployment-flask-master/Deployment-flask-master/Tweets.csv')
    train=tweets.loc[:,["text"]]
    tokenizer = Tokenizer(num_words = 4000)
    tokenizer.fit_on_texts(train["text"])
    sequence = tokenizer.texts_to_sequences(text)
    max_seq_len = 1000
    padded_seq = pad_sequences(sequence , maxlen = max_seq_len )
    output=model.predict(padded_seq)
    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output[1]))

if __name__ == "__main__":
    app.run(debug=True)