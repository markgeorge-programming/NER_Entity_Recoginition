from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)


MAX_LENGTH = 20
TAG_MAP = {0: "O", 1: "COLOR", 2: "PRODUCT", 3: "DATE", 4: "LOCATION"}

product_db={"sneakers","shirts","shorts","boots","coat","hat","tshirt","shoes"}

LOCATION_DB = {
    "london", "paris", "new york", "dubai", "tokyo", "berlin",
    "usa", "uk", "france", "germany", "japan",
    "egypt", "cairo", "giza", "alexandria", "alex",
    "helwan", "maadi", "zamalek", "nasr city", "6th of october",
    "sharm", "hurghada", "dahab"
}
color_db = {
    "red", "blue", "green", "yellow", "black", "white", "orange",
    "purple", "pink", "brown", "grey", "gray", "silver", "gold"
}
date_db={"today", "yesterday", "tomorrow", "tonight",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "weekend", "morning", "evening", "night"}

"""
ANIME_DB = {
    "jujutsu", "kaisen", "gojo", "sukuna", "itadori",
    "naruto", "sasuke", "sakura",
    "bleach", "ichigo",
    "luffy", "zoro", "one piece",
    "blue lock", "isagi", "rin", "bachira"
}
"""
model = tf.keras.models.load_model('ner_high_accuracy.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        text = request.form.get('sentence', '')

        # Preprocess
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post')

        # Predict
        pred = model.predict(padded)
        ids = np.argmax(pred, axis=-1)[0]

        # Format Output
        words = text.split()
        for i, word in enumerate(words):
            if i < MAX_LENGTH:
                # Get the confidence score
                raw_conf = np.max(pred[0][i])

                #Gatekeeper Logic for low accuracies (hallucination error handling)
                if raw_conf > 0.79:
                    tag = TAG_MAP.get(ids[i], "O")
                elif word.lower() in product_db:
                    tag = "PRODUCT"
                    raw_conf = 1.0
                elif word.lower() in LOCATION_DB:
                    tag = "LOCATION"
                    raw_conf = 1.0
                elif word.lower() in date_db:
                    tag = "DATE"
                    raw_conf = 1.0
                elif word.lower() in color_db:
                    tag="COLOR"
                    raw_conf = 1.0
                else:
                    tag = "O"

                # Append results
                results.append({
                    "word": word,
                    "tag": tag,
                    "conf": f"{raw_conf:.2%}"  # Show the raw score so you know
                })

    return render_template('index.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)