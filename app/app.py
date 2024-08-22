from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from functions import *

app = Flask(__name__)

with open('model/spam-vs-ham.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Capture form data
        subject = request.form['input-subject']
        email = request.form['input-email']

        # Convert form data to a singular row in pandas
        data = {'subject': [subject], 'email': [email]}
        df = pd.DataFrame(data)
        
        # Apply data transformations for modeling
        df = df_to_lower(df)
        words_of_interest = ['3d', 'font', 'br', 'content']
        woi_df = pd.DataFrame(words_in_texts(words_of_interest, df['email']), columns=words_of_interest)
        df = transformations(df)
        df = pd.concat([df, woi_df], axis=1)
        X = df[['html_tags', 'body_characters', 'body_length', 'exclamations', 'is_reply', 'has_ip', '3d', 'font', 'br', 'content']]

        # Make prediction based on input subject line and email body
        prediction = model.predict(X)[0]
        pred_to_result = {0:'ham', 1:'spam'}
        return render_template('index.html', 
                               spam_or_ham=pred_to_result[prediction], 
                               html_tags=X['html_tags'].iloc[0],
                               subject_len=df['subj_length'].iloc[0],
                               body_len=X['body_length'].iloc[0],
                               body_char=X['body_characters'].iloc[0],
                               excl_count=X['exclamations'].iloc[0],
                               subject=subject,
                               email=email)

    return render_template('index.html', 
                           spam_or_ham=' ', 
                           html_tags=' ',
                           subject_len=' ',
                           body_len=' ',
                           body_char=' ',
                           excl_count=' ',
                           subject='',
                           email='')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

if __name__ == '__main__':
    app.run(debug=True)