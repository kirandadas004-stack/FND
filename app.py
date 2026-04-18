from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    vec = vectorizer.transform([news])
    result = model.predict(vec)[0]

    if result == 1:
        return render_template("index.html", prediction="Real News ✅")
    else:
        return render_template("index.html", prediction="Fake News ❌")

if __name__ == "__main__":
    app.run(debug=True)
