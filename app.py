import os
import urllib.request
#from app import app
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

from flask import Flask, request
from keras_vggface.vggface import VGGFace
from keras.models import load_model
import os
import json
import cv2
from img_processing import face_matching, get_model, crop_image, cos_similarity, extract_face

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER="/home/deeplearningcv/Downloads/fraud_detection/images"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SQLALCHEMY_DATABASE_URI']= 'sqlite:////home/deeplearningcv/Downloads/fraud_detection_apis/filestorage.db'
db=SQLAlchemy(app)




class FileUpload(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    name=db.Column(db.String(300))
    data=db.Column(db.LargeBinary)



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upl', methods=['POST'])

def upload_file():
    if 'file1' not in request.files or 'file2' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp

    file1=request.files['file1']
    file2=request.files['file2']
    


    if file1.filename =='' or file2.filename=='':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if (file1 and allowed_file(file1.filename)) and (file2 and allowed_file(file2.filename)):
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)


        file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))

        img_path1=os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        img_path2=os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        res = face_matching(img_path1, img_path2)
        resp_data = {"match": res}

        fileup1=FileUpload(name=filename1,data=file1.read())
        fileup2=FileUpload(name=filename2,data=file2.read()) 
        db.session.add(fileup1)
        db.session.add(fileup2)
        db.session.commit()

        return jsonify(resp_data)
    else:
        resp = jsonify({'message' : 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp




if __name__=='__main__':
     app.run("localhost",9999,debug=True)
