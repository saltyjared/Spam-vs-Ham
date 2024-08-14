from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

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
        data = {'subject':subject, 'email':email}
        df = pd.DataFrame(data)
        print(df)

    return render_template('index.html', x='placeholder')

if __name__ == '__main__':
    app.run(debug=True)