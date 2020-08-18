from flask import Flask, render_template, request, session
import sys
import uuid

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
    url = request.form['url']
    result = 'result'
    return result

if __name__ == '__main__':
    app.run()
