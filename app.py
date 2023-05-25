from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import random

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

# Load and preprocess the data
f = open('chatbot.txt', 'r',encoding="utf8")
raw_doc = f.read()
raw_doc = raw_doc.lower()
sent_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def response(user_response):
    
    robo1_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    
    tfidf = TfidfVec.fit_transform(sent_tokens)  # Fit the vectorizer with existing data
    user_tfidf = TfidfVec.transform([user_response])  # Transform user input
    
    vals = cosine_similarity(user_tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo1_response = robo1_response + "I am sorry! I don't understand you"
    else:
        robo1_response = robo1_response + sent_tokens[idx]

    return robo1_response


GREET_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREET_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greet(sentence):
    for word in sentence.split():
        if word.lower() in GREET_INPUTS:
            return random.choice(GREET_RESPONSES)

@app.route('/home')
def serve_frontend():
    return app.send_static_file('front.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    global sent_tokens, word_tokens
    
    user_input = request.json['user_input']
    user_input = user_input.lower()

    print(user_input)
    
    if user_input == 'bye':
        bot_response = "Goodbye! Take care <3"
    elif user_input == 'thanks' or user_input == 'thank you':
        bot_response = "You're welcome!"
    elif greet(user_input) is not None:
        bot_response = greet(user_input)
    else:
        sent_tokens.append(user_input)
        word_tokens = word_tokens + nltk.word_tokenize(user_input)
        bot_response = response(user_input)
        sent_tokens.remove(user_input)

    return jsonify({'bot_response': bot_response})

debug = False

app.static_folder = ''
if __name__ == '__main__':
    if not debug:
        from waitress import serve
        try:
            print("Starting server...")
            serve(app, host='0.0.0.0', port=8000)
        except Exception as e:
            print("An error occurred:", e)
    else:
        user_input = "điểm thi ngành thiết kế đồ hoạ khối a là ggi ??"
        
        print(user_input)
        
        sent_tokens.append(user_input)
        word_tokens = word_tokens + nltk.word_tokenize(user_input)
        bot_response = response(user_input)
        sent_tokens.remove(user_input)
        
        print(bot_response)
