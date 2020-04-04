import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.get_default_graph
sess = tf.Session(config=config)

from tensorflow.python.keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img as limg
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
import librosa
from model import define_gan
from wav2mel_mel2wav.pghi_spec2wav import wav2spec, spec2wav
import numpy as np
import soundfile as sf

def load_img(filename):
	pixels = limg(filename, grayscale=True, target_size=(256,256))
	pixels = img_to_array(pixels)
	pixels = (pixels - 127.5) / 127.5
	pixels = expand_dims(pixels, 0)
	return pixels
	
def img2spec(img_path, out_path, model_path='trainlog_datasmall_W_8k_256x256/checkpoint/epoch_090.h5'):
	img = load_img(img_path)
	print(img.shape)
	model = load_model(model_path)
	print('Loaded model from:', model_path)
	spec = model.predict(img)[0,:,:,0]
	print(spec.shape)
	audio_signal = spec2wav(spec)
	#maxv = np.iinfo(np.int16).max
	#audio_signal = (audio_signal * maxv).astype(np.int16)
	#librosa.output.write_wav(out_path, audio_signal, sr=8000)
	sf.write(out_path, audio_signal, 8000, subtype='PCM_16')
	
#img2spec('mnist_png/testing/0/3.png','zero.wav','trainlog_datasmall_8000_256x256/checkpoint/epoch_100.h5')
	

	