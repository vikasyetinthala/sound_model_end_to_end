from flask import Flask,jsonify,request
import random
import os
import json
from sound_classification_service import sound_classification_service


app=Flask(__name__)

@app.route("/predict",methods=['POST'])
def predict():
    audio_file=request.files["file"]
    file_name=str(random.randint(1,100000))
    audio_file.save(file_name)
    scs=sound_classification_service()
    keyword=scs.predict(file_name)
    os.remove(file_name)
    data={'keyword':keyword}
    return jsonify(data)


if __name__=="__main__":
    app.run(debug=False)

