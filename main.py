from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
from flask_cors import CORS
from flask_restful import Resource, Api
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, concatenate, Bidirectional, Dot, Activation, RepeatVector, Multiply, Lambda
import h5py
import requests
import re

# Initialize Flask App
app = Flask(__name__)
api = Api(app)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model and tokenizer paths
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, 'model', 'resnet50.weights.h5')
tokenizer_path = os.path.join(script_dir, 'model', 'tokenizer.pkl')

# Load model and tokenizer
with h5py.File(model_path, 'r') as f:
    print(f.keys())

max_caption_length = 34
vocab_size = 8768

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
fe2_projected = RepeatVector(max_caption_length)(fe2)
fe2_projected = Bidirectional(LSTM(256, return_sequences=True))(fe2_projected)

# Sequence feature layers
inputs2 = Input(shape=(max_caption_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = Bidirectional(LSTM(256, return_sequences=True))(se2)

class AttentionContext(tf.keras.layers.Layer):
    def call(self, inputs):
        attention_scores, se3 = inputs
        return tf.einsum('ijk,ijl->ikl', attention_scores, se3)

class ContextVector(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

attention = Dot(axes=[2, 2])([fe2_projected, se3])
attention_scores = Activation('softmax')(attention)
attention_context = AttentionContext()([attention_scores, se3])
context_vector = ContextVector()(attention_context)

# Decoder model
decoder_input = concatenate([context_vector, fe2], axis=-1)
decoder1 = Dense(256, activation='relu')(decoder_input)
outputs = Dense(vocab_size, activation='softmax')(decoder1)

# Create the model
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.load_weights(model_path)

# Load the tokenizer
with open(tokenizer_path, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

extract_model = ResNet50()
extract_model = Model(inputs=extract_model.inputs, outputs=extract_model.layers[-2].output)

def get_word_from_index(index, tokenizer):
    return next((word for word, idx in tokenizer.word_index.items() if idx == index), None)

def predict_caption(model, image_features, tokenizer, max_caption_length):
    caption = 'startseq'
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length, padding='post')
        yhat = model.predict([image_features, sequence], verbose=0)
        predicted_index = np.argmax(yhat)
        predicted_word = get_word_from_index(predicted_index, tokenizer)
        caption += " " + predicted_word
        if predicted_word is None or predicted_word == 'endseq':
            break
    return caption

def generate_caption_new_image(image_name):
    img_path = os.path.join(script_dir, UPLOAD_FOLDER, image_name)
    image = Image.open(img_path)
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    image_feature = extract_model.predict(image, verbose=0)
    return predict_caption(model, image_feature, tokenizer, max_caption_length)

class ImageUpload(Resource):
    def post(self):
        if 'image' not in request.files:
            return {'error': 'No image file provided'}, 400
        image_file = request.files['image']
        if image_file.filename == '':
            return {'error': 'No selected file'}, 400

        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)
        try:
            caption = generate_caption_new_image(filename)
        except Exception as e:
            return {'error': f'Prediction error: {str(e)}'}, 500
        return jsonify({'caption': caption, 'file_path': filepath})

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = {'1_image': request.form.get('1_image'),'1_tone': 'fun', '1_additionalInfo': '','1_language': 'en-US', '0': '[\"$K1\"]'}
        headers = {'next-action': 'c3d48a60acb61095f31e8af62b917dc2fe6d63a0'}
        target_url = 'https://imagecaptiongenerator.com/'
        response = requests.post(target_url, headers=headers,data=data)
        try:
            response_data = response.text
            cleaned_captions = response_data.replace(r"^\d+\.\s*", "")
            cleaned_captions = cleaned_captions.replace(r"[^\x20-\x7F]+", "")
            captions = [line.strip() for line in cleaned_captions.split("\\n")[1:-1] if line.strip()]
        except ValueError as e:
            print("JSON Parse Error:", e)
            return jsonify({"error": "Failed to parse response JSON", "raw_response": response.text}), 500

        return jsonify({
            "status_code": response.status_code,
            "data": captions
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

api.add_resource(ImageUpload, '/predict')

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--port', type=str, default=1412, help='port to listen on')
    args = vars(parser.parse_args())
    app_port = args['port']
    app.run(host='0.0.0.0', port=app_port)