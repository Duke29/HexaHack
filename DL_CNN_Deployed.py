import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
model = load_model('spam_detection_cnn.h5')
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(pd.read_csv('spam_ham_dataset.csv', usecols=[2], encoding='latin-1')['text'])
max_len = 100
text = input("Enter a message: ")
text = [text]
text = tokenizer.texts_to_sequences(text)
text = pad_sequences(text, padding='post', maxlen=max_len)
predictions = model.predict(text)
if predictions[0] > 0.1:
    print("This is a spam message.")
else:
    print("This is not a spam message.")
print("Risk Factor:"+predictions[0])
