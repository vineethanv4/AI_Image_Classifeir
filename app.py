from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

app = Flask(__name__)

dic = {0 : 'Fake Image', 1 : 'Real Image'}


def predict_label(img_path):
	gen=ImageDataGenerator()
	data = [[img_path, "REAL"]]
	# Create the pandas DataFrame
	test_df = pd.DataFrame(data, columns=['filepaths', 'labels'])
	# write try and catch
	try:
		test_gen=gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=(32, 32), class_mode= 'categorical', color_mode='rgb', shuffle=False, batch_size = 1)
		prediction = model.predict(test_gen, verbose=1)
		if prediction[0][0] > prediction[0][1]:
			return "Fake Image"
		return "Real Image"
	except:
		return "Invalid Image Format"



def F1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

model = load_model('Real vs Fake Images-2-(32 X 32)- 95.92.h5', {"F1_score": F1_score})

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		print(img.filename)

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)