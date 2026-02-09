from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

# Constants (Must match training script)
MAX_LENGTH = 20
TAG_MAP = {0: "O", 1: "COLOR", 2: "PRODUCT", 3: "DATE", 4: "LOCATION"}

# Load assets ONCE at startup
model = tf.keras.models.load_model('ner (1).h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        text = request.form.get('sentence', '')

        # 1. Preprocess
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post')

        # 2. Predict
        pred = model.predict(padded)
        ids = np.argmax(pred, axis=-1)[0]

        # 3. Format Output
        words = text.split()
        for i, word in enumerate(words):
            if i < MAX_LENGTH:
                tag = TAG_MAP.get(ids[i], "O")
                conf = np.max(pred[0][i])
                results.append({"word": word, "tag": tag, "conf": f"{conf:.2%}"})

    return render_template('index.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)