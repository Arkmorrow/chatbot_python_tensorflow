import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
from tensorflow.python.framework import ops
import random
import json
import pickle
import os
import time

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

def bag_of_words(s, words):
  bag = [0 for _ in range(len(words))]

  s_words = nltk.word_tokenize(s)
  s_words = [stemmer.stem(word.lower()) for word in s_words]

  for se in s_words:
      for i, w in enumerate(words):
          if w == se:
              bag[i] = 1
          
  return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:

        #get msg input path
        path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
        path+= "/Discord Bot/msg.txt"
        msg_file = open(path,"r")
        input_msg = ""
        if msg_file.read(1) == '-':
            while True:
                c = msg_file.read(1)
                if not c:
                    break
                input_msg+= c
            
            msg_file.close()

            #setting the result
            print(input_msg)
            results = model.predict([bag_of_words(input_msg, words)])
            results_index = numpy.argmax(results)
            tag = labels[results_index]

            print(results)
            print(results_index)

            #write back to the msg file
            new_file = open(path,"w")

            if results_index < 0:
                print("Sorry, I can understand you. Please train me.")
                new_file.write("Sorry, I can understand you. Please train me.")
                new_file.close()

            else:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']

                new_file.write(random.choice(responses))
                new_file.close()
                time.sleep(1)
                msg_file = open(path,"r")
                print(msg_file.read())
                msg_file.close()
            
        

chat()