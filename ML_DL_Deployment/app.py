from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return("XXXXXXX")

@app.route('/page2')
def page():
    return("XXXXXXX2")

def main():
    app.run(debug=True)

if __name__ == "__main__":
    main()