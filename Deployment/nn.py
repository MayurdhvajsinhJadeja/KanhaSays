import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
lemmatizer = WordNetLemmatizer()
words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('Deployment/gita_intents.json', encoding='utf-8').read()
intents = json.loads(data_file)
for intent in intents['intents']:
    for pattern in intent['patterns']:

        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)
pickle.dump(words,open('texts.pkl','wb'))
pickle.dump(classes,open('labels.pkl','wb'))
# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=280, batch_size=5, verbose=1)
model.save('model.h5', hist)
print("model created")


# import json
# import pickle
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout, Embedding, Flatten
# from keras.optimizers import SGD
# from keras.preprocessing.text import Tokenizer
# from keras_preprocessing.sequence import pad_sequences

# from sklearn.preprocessing import LabelEncoder
# import random

# # load data
# data_file = open('Deployment/gita_intents.json', encoding='utf-8').read()
# intents = json.loads(data_file)

# # extract patterns and labels
# patterns = []
# labels = []
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         patterns.append(pattern)
#         labels.append(intent['tag'])

# # tokenize patterns
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(patterns)
# word_index = tokenizer.word_index

# # convert patterns to sequences
# sequences = tokenizer.texts_to_sequences(patterns)

# # pad sequences to max length
# maxlen = max([len(seq) for seq in sequences])
# padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')

# # convert labels to one-hot encoding
# label_encoder = LabelEncoder()
# label_encoder.fit(labels)
# encoded_labels = label_encoder.transform(labels)
# num_classes = len(label_encoder.classes_)
# onehot_labels = np.zeros((len(encoded_labels), num_classes))
# for i, label in enumerate(encoded_labels):
#     onehot_labels[i][label] = 1

# # shuffle data
# data = list(zip(padded_sequences, onehot_labels))
# random.shuffle(data)
# padded_sequences, onehot_labels = zip(*data)

# # split data into train and test sets
# split_idx = int(len(padded_sequences) * 0.8)
# train_x = np.array(padded_sequences[:split_idx])
# train_y = np.array(onehot_labels[:split_idx])
# test_x = np.array(padded_sequences[split_idx:])
# test_y = np.array(onehot_labels[split_idx:])

# # load pre-trained word embeddings
# embedding_dim = 100
# embedding_index = {}
# with open('Deployment/glove.6B.100d.txt', encoding='utf-8') as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embedding_index[word] = coefs

# # create embedding matrix
# num_words = len(word_index) + 1
# embedding_matrix = np.zeros((num_words, embedding_dim))
# for word, i in word_index.items():
#     embedding_vector = embedding_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector

# # define model architecture
# model = Sequential()
# model.add(Embedding(num_words, embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# # compile model
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# # train model
# hist = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=250, batch_size=5, verbose=1)
# model.save('chatbot_model.h5', hist)
# print("Model trained and saved to file.")