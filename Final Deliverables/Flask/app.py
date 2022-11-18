# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 09:20:07 2022

@author: Admin
"""

from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.models import load_model
from flask import Flask, request, render_template,url_for,redirect
from werkzeug.utils import secure_filename
from keras.models import model_from_json

global graph
graph=tf.get_default_graph()

app = Flask(__name__)

json_file = open('final_model.json','r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('final_model.h5')

print('Model loaded. Chesk http://127.0.0.1:5000/')

@app.route('/',methods=['GET'])
def index():
    return render_template('digital.html')

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        f = request.files['image']
        print(type(f))
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'static','uploads', f.filename)
        f.save(file_path)
        print(file_path)
        img = image.load_img(file_path, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        with graph.as_default():
            preds = loaded_model.predict_classes(x)
            
        print(preds)
            
        found = ["The great Indian bustard (Ardeotis nigriceps) or Indian bustard, is a bustard found on the Indian subcontinent. A large bird with a horizontal body and long bare legs, giving it an ostrich like appearance, this bird is among the heaviest of the flying birds.",
                 "The spoon-billed sandpiper (Calidris pygmaea) is a small wader which breeds on the coasts of the Bering Sea and winters in Southeast Asia.",
                 "Amorphophallus titanum, the titan arum, is a flowering plant in the family Araceae. It has the largest unbranched inflorescence in the world. The inflorescence of the talipot palm, Corypha umbraculifera, is larger, but it is branched rather than unbranched. A. titanum is endemic to rainforests on the Indonesian island of Sumatra.",
                 "lady’s slipper, (subfamily Cypripedioideae), also called lady slipper or slipper orchid, subfamily of five genera of orchids (family Orchidaceae), in which the lip of the flower is slipper-shaped. Lady’s slippers are found throughout Eurasia and the Americas, and some species are cultivated.",
                 "Pangolins, sometimes known as scaly anteaters,[5] are mammals of the order Pholidota, Pangolins have large, protective keratin scales, similar in material to fingernails and toenails, covering their skin; they are the only known mammals with this feature. They live in hollow trees or burrows, depending on the species. Pangolins are nocturnal, and their diet consists of mainly ants and termites, which they capture using their long tongues. ",
                 "The Seneca white deer are a rare herd of deer living within the confines of the former Seneca Army Depot in Seneca County, New York. These deer are not albino, but instead have leucism, which is an abnormal genetic condition that carries a set of recessive genes for all-white coats."]
        print('uploads/'+f.filename)
        text=found[preds[0]]
        return render_template('digital.html', msg=text,img=f.filename)

@app.route('/uploads/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

    
if __name__=='__main__':
    app.run(threaded = False)