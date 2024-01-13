from flask import Flask, render_template, request
from minmax import MinMaxNormalization
from classification import perform_svm_classification
import pickle
from svm import SVM

MinMax = MinMaxNormalization()
with open("models/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

app = Flask(__name__, static_url_path='/static')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        review_helpful = int(request.form.get('helpful'))
        sentiment = float(request.form.get('sentiment'))
        subjectivity = float(request.form.get('subjectivity'))
        word_count = int(request.form.get('words'))
        noun_count = int(request.form.get('noun'))
        adj_count = int(request.form.get('adj'))
        verb_count = int(request.form.get('verb'))
        adv_count = int(request.form.get('adv'))
        authenticity = float(request.form.get('authentic'))
        at = int(request.form.get('at'))

    new_data = MinMax.normalize(review_helpful, sentiment, subjectivity, word_count, noun_count, adj_count, verb_count, adv_count, authenticity, at)

    predictions = model.predict(new_data)
    return render_template("index.html", predictions=predictions, helpful=review_helpful, sentiment=sentiment, subjectivity=subjectivity, words=word_count, noun=noun_count, adj=adj_count, verb=verb_count, adv=adv_count, authentic=authenticity, at=at)

@app.route("/train", methods=["GET", "POST"])
def train():
    TP = TN = FP = FN = accuracy = precision = recall = f1_score = 0
    if request.method == "POST":
        file_path = request.files['file']
        split_ratio = float(request.form['split'])
        c_value = float(request.form['C'])

        TP, TN, FP, FN, accuracy, precision, recall, f1_score = perform_svm_classification(file_path, split_ratio, c_value)

    return render_template("train.html", TP=TP, TN=TN, FP=FP, FN=FN, accuracy=accuracy, precision=precision, recall=recall, f1_score=f1_score)

if __name__=="__main__":
    app.run(debug=True)