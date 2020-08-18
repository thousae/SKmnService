from flask import Flask, render_template, request, session
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

@app.route('/get-result', methods=['POST'])
def show_result():
    link = request.form['url']
    if not link:
        return render_template('show_result.html', result=None)
    result = None
    return render_template('show_result.html', result=result)

if __name__ == '__main__':
    app.run()
