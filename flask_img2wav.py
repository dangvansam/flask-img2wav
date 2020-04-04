import os
import requests
from flask import Flask, escape, request, render_template, session
#from src.add_noise import add_noise_one
from predict_img2spec import img2spec
import random
app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html', filename='', img_filename='')
@app.route('/predict', methods=['POST','GET'])
def predict():
	if request.method == 'POST':
		file = request.files['file']
		filename = file.filename
		print(filename)
		file_path = 'static/upload/'+filename
		file.save(file_path)
		data = requests
		pred_file_path = 'static/upload/'+filename.split('.')[0] + '_pred.wav'
		img2spec(file_path, pred_file_path)
	return render_template('index.html', filename=pred_file_path, img_filename=file_path)
	
if __name__ == "__main__":
    app.debug = True
    app.secret_key = 'dangvansam'
    app.run(port='4040')