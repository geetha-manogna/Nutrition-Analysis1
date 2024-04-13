import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph
graph = tf.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("nutrition.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
            
            
            print("prediction",preds)
            
        index = ['Apple','Banana','brinjal','capsicum','cauliflower','corn','grapes','lady finger','Mango','orange','pineapple','watermelon']
        if(str(index[preds[0]])=="Apple"):
        	text = "the predicted Fruit or vegetable is : " + str(index[preds[0]]) + " and the nutrition values of "+str(index[preds[0]])+"  are-- calories=52 g,protein=0.3 g,   total_fat=0.2 g, total_carbo=14 g"
        elif(str(index[preds[0]])=="Banana"):
        	text = "the predicted Fruit or vegetable is : " + str(index[preds[0]]) + " and the nutrition values of "+str(index[preds[0]])+"  are-- calories=89 g,protein=1.1 g,   total_fat=0.3 g, total_carbo=23 g"
        elif(str(index[preds[0]])=="brinjal"):
        	text = "the predicted Fruit or vegetable is : " + str(index[preds[0]]) + " and the nutrition values of "+str(index[preds[0]])+"  are-- calories=25 g,protein=1 g,   total_fat=0.2 g, total_carbo=6 g"	
        elif(str(index[preds[0]])=="capsicum"):
        	text = "the predicted Fruit or vegetable is : " + str(index[preds[0]]) + " and the nutrition values of "+str(index[preds[0]])+"  are-- calories=40 g,protein=1.9 g,   total_fat=0.2 g, total_carbo=9 g"
        elif(str(index[preds[0]])=="cauliflower"):
        	text = "the predicted Fruit or vegetable is : " + str(index[preds[0]]) + " and the nutrition values of "+str(index[preds[0]])+"  are-- calories=25 g,protein=1.9 g,   total_fat=0.3 g, total_carbo=5 g"
        elif(str(index[preds[0]])=="corn"):
        	text = "the predicted Fruit or vegetable is : " + str(index[preds[0]]) + " and the nutrition values of "+str(index[preds[0]])+"  are-- calories=44 g,protein=1.6 g,   total_fat=0.8 g, total_carbo=7.8 g"
        elif(str(index[preds[0]])=="grapes"):
        	text = "the predicted Fruit or vegetable is : " + str(index[preds[0]]) + " and the nutrition values of "+str(index[preds[0]])+"  are-- calories=67 g,protein=0.6 g,   total_fat=0.4 g, total_carbo=17 g"
        elif(str(index[preds[0]])=="lady finger"):
        	text = "the predicted Fruit or vegetable is : " + str(index[preds[0]]) + " and the nutrition values of "+str(index[preds[0]])+"  are-- calories=60 g,protein=0.8 g,   total_fat=0.4 g, total_carbo=15 g"
        elif(str(index[preds[0]])=="Mango"):
        	text = "the predicted Fruit or vegetable is : " + str(index[preds[0]]) + " and the nutrition values of "+str(index[preds[0]])+"  are-- calories=60 g,protein=0.8 g,   total_fat=0.4 g, total_carbo=15 g"
        elif(str(index[preds[0]])=="orange"):
        	text = "the predicted Fruit or vegetable is : " + str(index[preds[0]]) + " and the nutrition values of "+str(index[preds[0]])+"  are-- calories=47 g,protein=0.9 g,   total_fat=0.1 g, total_carbo=12 g"		
        elif(str(index[preds[0]])=="pineapple"):
        	text = "the predicted Fruit or vegetable is : " + str(index[preds[0]]) + " and the nutrition values of "+str(index[preds[0]])+"  are-- calories=50 g,protein=0.5 g,   total_fat=0.1 g, total_carbo=13 g"
        else:
        	text = "the predicted Fruit or vegetable is : " + str(index[preds[0]]) + " and the nutrition values of "+str(index[preds[0]])+"  are-- calories=47 g,protein=0.9 g,   total_fat=0.2 g, total_carbo=8 g"

    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        
        
        
    
    
    