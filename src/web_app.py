import json
import graphlab as gl
import socket
import time
import pandas as pd
from flask import Flask, jsonify, render_template, request
import cPickle as pickle
from pymongo import MongoClient
import requests
import json
import numpy as np
import pandas as pd
from datapoint_pipeline import open_cPickle_file, extract_and_filter_datapoint_text

app = Flask(__name__)
PORT = 8080
model = open_cPickle_file('Fitted_Model_AdaBoostClassifier_Style')
#model = gl.load_model('')
#rf = gl.load_model('')

#post = pd.read_csv('')
#columns = post.columns
#X = post[columns]
#X = X.drop_duplicates()

@app.route('/home')
def index():
    return render_template('~/Desktop/lattes/index.html')


@app.route('/app')
def webapp():
    return render_template('lattes/web_app.html')

@app.route('/prediction', methods =['POST'])
def predict_gender():

    user_text = [request.form['user_input']]
    prediction = extract_and_filter_datapoint_text(user_text)

    if prediction == 'feminine':
        text_len = 'shorter'
        word_use = 'greater'
        sent_len = 'longer'
        sent_var = 'more varied'
        quotes = 'more'
        polar = 'greater'
        formal = 'less' 
    elif predicition == 'masculine':
        text_len = 'longer'
        word_use = 'weaker'
        sent_len = 'shorter'
        sent_var = 'less varied'
        quotes = 'less'
        polar = 'less'
        formal = 'greater' 

    detailed_analysis = "This text shows a {{prediction}} style of writing, which is usually expressed by stylistic \
        features that suggest a {{word_use}} diversity of vocabulary, {{sent_len}} and {{sent_var}} \
        sentence lengths, {{text_len}} article length and use of {{quotes}} quotations. \
        Additionally, {{prediction}} writing style tends to exhibit {{polar}} polarity \
        and {{formal}} formality." 

    return render_template('lattes/web_app.html', detailed_analysis=detailed_analysis, prediction=prediction)

# @app.route('/about_us')
# def about_us():
#     return render_template('about_us.html')

if __name__ == '__main__':


   # Start Flask app
    app.run(host='0.0.0.0', port=8080, debug=True)


# if __name__ == '__main__':
#     # Register for pinging service
#     ip_address = socket.gethostbyname(socket.gethostname())
#     print "attempting to register %s:%d" % (ip_address, PORT)
#     register_for_ping(ip_address, str(PORT))
#
#     # Start Flask app
#     app.run(host='0.0.0.0', port=PORT, debug=True)
