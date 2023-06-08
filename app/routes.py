from app import app
from app.lang_dectect import LanguageDetector
from flask import render_template, request, jsonify
import os
import csv
import sys

model_path = "models/saved_model/simple_mlp_novectorize.h5"
vectorizer_path = "models/vectorizer"


@app.route('/', methods=['GET', 'POST'])
def index():
    print(request.method)
    if request.method == 'POST':
        if request.form.get('Detect Language') == 'Detect Language':
            text = request.form['text']
            detector = LanguageDetector(model_path, vectorizer_path)
            language = detector.detect_language(text)
            return render_template('index.html', text=text, language=language)
        return render_template('index.html')
    return render_template('index.html')


@app.route('/detect_language', methods=['POST'])
def detect_language():
    print(request.method)
    if request.method == 'POST':
        text = request.form['text']
        detector = LanguageDetector(model_path, vectorizer_path)
        language = detector.detect_language(text)
        return render_template('index.html', text=text, language=language)
    return render_template('index.html')


@app.route('/store_data', methods=['POST'])
def store_data():
    if request.method == 'POST':
        text = request.form['text']
        language = request.form['language']

        print("Text: ", text, file=sys.stderr)
        print("Language: ", language, file=sys.stderr)

        csv_file_path = os.path.join(app.root_path, 'data.csv')

        with open(csv_file_path, 'a', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([text, language])

            print("Wrote data to csv file", file=sys.stderr)

        

        return render_template('index.html', text=None, language=None)



