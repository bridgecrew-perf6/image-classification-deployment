from flask import Flask, redirect, url_for, request, render_template
import pickle,keras,cv2
import os
import glob
import re
import numpy as np

VGG_model=keras.models.load_model("vgg16.h5", custom_objects=None, compile=True, options=None)
model = pickle.load(open('RF_model.pkl','rb'))
train_labels=['cat','dog']

app = Flask(__name__)
UPLOAD_FOLDER="upload"


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')




@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method=="POST":
        image_file= request.files["image"]
        if image_file:
            image_location= os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            
            
            image_file.save(image_location)
            
            x=image_location.split('\\')
            
            img="upload"+"/"+x[1]
            
            img = cv2.imread(img)      
            img = cv2.resize(img, (256, 256))
            input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
            input_img_feature=VGG_model.predict(input_img)
            
            input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
            
            prediction_RF = model.predict(input_img_features)[0]
            
            result=train_labels[prediction_RF]
            
            print(result)
            return result
    return None

if __name__ == "__main__":
    app.run()