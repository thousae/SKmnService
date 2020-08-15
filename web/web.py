from flask import Flask, render_template, request
import sys

if '-d' in sys.argv:
    DEBUG = True
else:
    DEBUG = False

app = Flask(__name__)
app.config.from_envvar('WEB_SETTINGS', silent=True)

@app.route('/')
def show_main():
    return render_template('show_result.html', result=None)

@app.route('/', methods=['POST'])
def show_result():
    link = request.form['urlbox']
    if not link:
        return render_template('show_result.html', result=None)
    result = None
    return render_template('show_result.html', result=result)

if __name__ == '__main__':
    app.run()
