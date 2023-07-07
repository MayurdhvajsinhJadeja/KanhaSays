import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Use the lemmatizer directly from WordNet corpus
def lemmatize_word(word):
    lemma = wordnet._morphy(word, wordnet.NOUN)
    if lemma is None:
        lemma = wordnet._morphy(word, wordnet.VERB)
    if lemma is None:
        lemma = wordnet._morphy(word, wordnet.ADJ)
    if lemma is None:
        lemma = wordnet._morphy(word, wordnet.ADV)
    if lemma is None:
        lemma = word
    return lemma


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return bag


# Update the predict_class function to use lemmatize_word
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    global result
    try:
        tag = ints[0]["intent"]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
    except:
        result = "I cannot understand this statement. Perhaps rephrase it or type it differently?"
    return result

model = load_model("./model.h5")
def chatbot_response(msg, model):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res



# s = "how to attain moksha?"
# print(chatbot_response(s))

from flask import Flask, render_template, request, flash

# import pymysql

# # MySQL configuration
# db = pymysql.connect(
#     host="localhost",
#     user="root",
#     password="",
#     database="feedback"
# )


app = Flask(__name__)
app.secret_key = "5636-d7b6-d647-dd45-e434-8551-f27b-680d"
# app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("/index.html")


@app.route("/ask", methods=["POST", "GET"])
def get_bot_response():
    que = request.form["question"]
    ans = chatbot_response(que, model)  # Pass the 'model' variable as an argument
    return render_template("/index.html", answer=ans, question=que)

# @app.route('/ask',methods=["POST","GET"])
# def feedback():
#     return render_template('/error.html')

# @app.route('/submit-feedback', methods=['POST'])
# def submit_feedback():
#     name = request.form['name']
#     email = request.form['email']
#     feedback = request.form['feedback']

#     # Insert feedback into database
#     cursor = db.cursor()
#     sql = "INSERT INTO feedback (name, email, feedback) VALUES (%s, %s, %s)"
#     values = (name, email, feedback)
#     cursor.execute(sql, values)
#     db.commit()

#     # Redirect user back to index page with success message
#     flash('Thank you for your feedback!')
#     return render_template('/index.html')

if __name__ == "__main__":
    app.debug = True
    app.run()
