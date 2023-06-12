# importing necessary libraries
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras
import random
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split

# initializing some variables
words=[]
classes = []
documents = []
ignore_words = ['?', '!']

# reading and loading data from the intents json file
data_file = open(r'.\intents.json', encoding='utf-8').read()
intents = json.loads(data_file)

# creating a corpus of documents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))

# print some statistics
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

# save the processed data as a pickle file
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)

# create a bag of words representation for each document and its intent
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
training = np.array(training, dtype=object)

# Split the data into training and test sets (80% training, 20% test)
train_data, test_data = train_test_split(training, test_size=0.2, random_state=42)

# Convert the training and test data into numpy arrays
train_data = np.array(train_data, dtype=object)
test_data = np.array(test_data, dtype=object)

# Split the training and test data into X (patterns) and Y (intents)
train_x = list(train_data[:, 0])
train_y = list(train_data[:, 1])
test_x = list(test_data[:, 0])
test_y = list(test_data[:, 1])
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
sgd = tf.keras.optimizers.legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Evaluate the model (run test data that has not been used to train the model in order to check the results)
# This function returns a loss value and an accuracy value.  Lower loss value means better predictions, accuracy is a % value with 100% being the best.
model.evaluate(test_x,test_y)  

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")

from keras.models import load_model

# this command loads the previously trained model
# in order to not train the model everytime, the code could be split here
# and just load in the model after the initial training
model = load_model('chatbot_model.h5')

#Load in the data files with the model
intents = json.loads(open('intents.json', encoding='utf8').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

#This function takes in a sentence and a trained model as inputs and returns 
# a list of intents and their corresponding probabilities based on the input sentence.
def predict_class(sentence, model):
    # Tokenizes and lemmatizes the input sentence using the bow function defined earlier,
    # and stores the resulting bag-of-words array in p.
    p = bow(sentence, words,show_details=False)
    # Uses the trained model to predict the probability distribution of the input sentence 
    # belonging to each of the possible intents.
    res = model.predict(np.array([p]))[0]
    # Defines an error threshold
    ERROR_THRESHOLD = 0.25
    # creates a new list called results that contains tuples of the form (index of intent, 
    # probability) for all intents with a probability greater than the threshold.
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    # Creates a new list called return_list that contains dictionaries of the form 
    # {"intent": <intent name>, "probability": <probability>} for each of the intents in results. 
    # The intent name is looked up in the classes list using the index stored in the results tuples.
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# The getResponse function takes two arguments: ints, which is a list of dictionaries returned 
# by the predict_class function, and intents_json, which is a dictionary containing the intents 
# and responses for the chatbot. 
def getResponse(ints, intents_json):
    # get the predicted intent tag
    tag = ints[0]['intent']
    # load the intents from the json file
    list_of_intents = intents_json['intents']
    # find the matching intent
    for i in list_of_intents:
        if(i['tag']== tag):
            # randomly select a response from the matching intent
            result = random.choice(i['responses'])
            break
    # return the selected response
    return result

# The chatbot_response function takes a string argument text representing the user input.
# It then uses the predict_class() and getResponse() functions to return a response for the chatbot.
def chatbot_response(text):
    # get the predicted intents and their probabilities
    ints = predict_class(text, model)
    # get a response for the predicted intents
    res = getResponse(ints, intents)
    # return the response
    return res


#Creating GUI with tkinter
import tkinter
from tkinter import *

# The rest of the code defines a graphical user interface (GUI) for a chatbot application. 
# The user can interact with the chatbot by typing in messages and sending them to the bot 
# using a "Send" button. The chatbot responds to the user's messages and the conversation 
# is displayed in a chat window.
def send():
    # This function gets the text entered by the user in the EntryBox, removes any extra 
    # whitespace, and then clears the EntryBox.
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        # If the user entered some text, the function enables the chat window for editing and 
        # inserts the user's message into the chat window with the tag "You:".
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        # It then sets the chat window's color and font.
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        # Calls the chatbot_response() function with the user's message as an argument 
        # to get a response from the chatbot.
        res = chatbot_response(msg)
        # he response is then inserted into the chat window with the tag "Bot:".
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        # Finally, the chat window is disabled for editing and the scrollbar 
        # is moved to the bottom of the window.
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

# This creates a new window with the title "Hello" and sets the size to 400x500 pixels. 
# The window is not resizable.
base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window.  This creates a text widget for the chat window with a white background, 
# 8 lines, 50 characters per line, and the Arial font. The widget is initially disabled for 
# editing.
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window.  This creates a scrollbar widget and binds it to the chat window 
# using the command option. The scrollbar is associated with the yview() method of the chat 
# window, which allows the user to scroll up and down through the chat history.
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message.  This creates a button labeled "Send" with a green background 
# and white text. The button is associated with the send() function using the command option.
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message.  This creates a text widget for the user to enter messages 
# into with a white background, 29 characters per line, 5 lines, and the Arial font. The user 
# may send messages by hitting the return key also.
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")

#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

# base.mainloop() starts the GUI application and runs the program indefinitely until the user 
# closes the window, clicks a button or takes any other action that generates an event that 
# causes the application to exit.
base.mainloop()