from flask import Flask,render_template
app = Flask(__name__)
@app.route('/')
def hello():
    return render_template('home.html')

@app.route('/enter.html')
def enter():
    return render_template('enter.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/login.html')
def login():
    return render_template('login.html')
if __name__ == '__main__':
    app.run(debug=True)
