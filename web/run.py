from flask import Flask, render_template, request
from stt.make_wav import make_wav
from stt.generate_text import generate_text
from body_to_title.body_to_title_textrank import SKMN
import sys
import uuid

sys.path.append('~/Desktop/SKmnService-body-to-title')

if '-d' in sys.argv:
    DEBUG = True
else:
    DEBUG = False

app = Flask(__name__)
app.config.from_envvar('WEB_SETTINGS', silent=True)
app.secret_key = uuid.uuid4().hex

results = []

@app.route('/')
def show_main():
    return render_template('index.html', result=None)

@app.route('/get-result', methods=['POST'])
def show_result():
    method = request.form['method']
    url = request.form['url']

    filepath = generate_text(make_wav(method, url))
    with open(filepath, 'r') as f:
        text = f.read()
        result = SKMN(text, 1)
    print(result)
    return result

if __name__ == '__main__':
    app.run()
