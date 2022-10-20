#IN THIS EXAMPLE WE TEST A DATASET FOR PREDICTING IF A TEXT IS SARCASTIC
#

#--------------------------------------------------------------------------
#------------------------- IMPORTS ----------------------------------------
#--------------------------------------------------------------------------
from ast import Str
from cmath import log
import json
import os
from statistics import mode
#extra html
import urllib.request
#get only text data from html
from bs4 import BeautifulSoup
#regex -> can be use for replacing a removing words, digits etc on a text
import re
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
#custom local function
import gen_model
from gen_model import model_generator

from typing import Union

from fastapi import FastAPI
from fastapi import Request

app = FastAPI()
#--------------------------------------------------------------------------
#---------------------- variables -----------------------------------------
#--------------------------------------------------------------------------
sentences = []
labels = []
vocab_size = 40000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 20000  # we take a short % from the existing lines in sarcasm.json
#get current path
currentPath = os.getcwd()

#*************************************************************************************************************************************************************************
#  Load training json data  - example https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json from https://rishabhmisra.github.io/publications/
#*************************************************************************************************************************************************************************
print(currentPath)
with open(currentPath + "/stereotype.json", 'r', encoding='cp437') as f:
    datastore = json.load(f)

for item in datastore:
    sentences.append(item['sentence_text'])
    labels.append(item['stereotype'])

#*************************************************************************************************************************************************************************
#                            Dividing data into TEST AND TRAINING DATA ******* !IMPORTANT -- !IMPORTANT - !IMPORTANT
#*************************************************************************************************************************************************************************

split = round(len(sentences) * 0.8)
training_sentences = sentences[:split]
testing_sentences = sentences[split:]
training_labels = labels[:split]
testing_labels = labels[split:]

#*************************************************************************************************************************************************************************
#                                                               TOKENIZE data

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences,
                                maxlen=max_length,
                                padding=padding_type,
                                truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,
                               maxlen=max_length,
                               padding=padding_type,
                               truncating=trunc_type)

#*************************************************************************************************************************************************************************
#*********** esto es para trabahar con los arreglos multiples de las sentencias creadas anteriormente
#*************************************************************************************************************************************************************************
# Need this block to get it to work with TensorFlow 2.x
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

#***********************************************************************************************************************************************************
#                               AQUI HACEMOS EMBEDING -    NEURAL NETWORK CODE                       VER OPCIONES
#***********************************************************************************************************************************************************
#OPCION 1
#SI YA TENEMOS UN MODELO DESCARGADO Y ENTREDADO , PODEMOS CARGAR EN LA VARIABLE MODEL
model = tf.keras.models.load_model(currentPath + "/model_trained")

#OPCION 2
#SI NO TE SE TIENE UN MODELO , PODEMOS CONSTRUIRLO Y ENTRENARLO DE 0 , LLAMANDO LA SIGUIENTE FUNCIO Y LUEGO , CARGAR ESE MODELO EN LA VARIABLE MODEL
# model_generator(vocab_size=vocab_size,
#                 embedding_dim=embedding_dim,
#                 max_length=max_length,
#                 training_padded=training_padded,
#                 training_labels=training_labels,
#                 testing_padded=testing_padded,
#                 testing_labels=testing_labels)

# model = tf.keras.models.load_model(currentPath +
#                                    "/text_clasification/model_trained")
#***********************************************************************************************************************************************************
#                                               FIN   AQUI HACEMOS EMBEDING -    NEURAL NETWORK CODE
#***********************************************************************************************************************************************************

#***********************************************************************************************************************************************************
#                            AQUI PODEMOS IMPRIMIR GRAFICAS DEL COMPORTAMIENTO DE LOS CICLOS
#***********************************************************************************************************************************************************
# .......ingrese codigo con matplotlib
#

#***********************************************************************************************************************************************************
#                                                       AQUI PROBAMOS EL ENTRENAMIENTO CON 2 ORACIONES
#***********************************************************************************************************************************************************
#LA PRIMERA TIENE DENOTACIONES SARCASTISCAS , Y LA SEGUNDA NO

# print(model.predict(padded))
#RESULADO
#[[9.6066731e-01]  ---> PRIMERA ORACION BIEN ACERTADA
#[2.8870056e-05]]  ---> IGUAL


#NOTA , DEBEMOS TOMAR EN CUENTA QUE LA INFORMACION NOS DARÁ NUMEROS ENTRE 0 Y 1 SI SE ACERCAN AL RESULTADO DESEADO
# ES DECIR ,
#SI UNA ORACION TIENE SARCASMO, será igual a 0
#SI UNA ORACION NO TIENE SARCASMO , será igual a 1
def predictor_fn(text):
    sentence = [
        "She has employees who regularly tell her what they think she's doing wrong and where she should best spend the company's money, without any knowledge of the financial statements. He doesn't think those same employees would feel free to give so much unsolicited advice and feedback to a male CEO.",
    ]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences,
                           maxlen=max_length,
                           padding=padding_type,
                           truncating=trunc_type)
    ndata = model.predict(padded)
    return ndata


@app.get("/")
def read_root():
    return {"text": "test"}


@app.get("/predictor/{_:path}")
def read_item(request: Request):
    #11 remove /predictor/  from url path
    print(request.url.path[11:])
    urlFile = urllib.request.urlopen(request.url.path[11:])
    #reading html
    html = urlFile.read()
    #converting html to text
    soup = BeautifulSoup(html, 'html.parser')
    #getting everything from html
    text = soup.find_all(text=True)

    #here we remove elements we dont want from html
    output = ''
    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head',
        'input',
        'script',
        # there may be more elements you don't want, such as "style", etc.
    ]
    #here we get only text from html
    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)


#*************************************************************************************************************************************************************************
#                                               PREPARING TEXT FOR IA
#*************************************************************************************************************************************************************************

#Remove all the special characters
    result = re.sub(r'\W+', ' ', output)
    #Remove all single characters
    result = re.sub(r'\s+[a-zA-Z]\s+', ' ', result)
    #Remove single characters from the start
    result = re.sub(r'\^[a-zA-Z]\s+', ' ', result)
    #Removing NUMBERS
    result = re.sub(r'[0-9]', '', result)
    #subsituting multiple spaces with single space
    result = re.sub(r'\s+', ' ', result)
    #Removing prefixed 'b'
    result = re.sub(r'^b\s+', '', result)

    #converting to lowercase
    result = result.lower()

    #*************************************************************************************************************************************************************************
    #                                               PREDICTING
    #*************************************************************************************************************************************************************************

    return {"text": (int(predictor_fn(result)[0]) * 10) / 100}