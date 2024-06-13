from flask import Flask, render_template, request, redirect, url_for
from rhyme_model import get_res

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/eng', methods=['GET', 'POST'])
def laneng():
    if request.method == "POST":
        text = request.form['lyrics']
        text = text.split(';')
        return render_template("engmodel.html", res=get_res(text, 'eng'))
    else:
        return render_template("engmodel.html")


@app.route('/rus', methods=['GET', 'POST'])
def langru():
    if request.method == "POST":
        text = request.form['lyrics']
        text = text.split(';')
        return render_template("rusmodel.html", res=get_res(text, 'rus'))
    else:
        return render_template("rusmodel.html")


if __name__ == '__main__':
    app.run()
