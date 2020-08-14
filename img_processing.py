from keras_vggface.vggface import VGGFace
from keras.models import load_model

def get_model():
  model=load_model("/home/deeplearningcv/Downloads/fraud_detection/model.h5")
  return model


from mtcnn import MTCNN
import numpy as np
import cv2
import matplotlib.pyplot as plt
#from keras_vggface.utils import preprocess_input

def crop_image(file_name,taget_size=(224,224)):
    

    img=cv2.imread(file_name)
    #img=img.astype('float32')
    detector=MTCNN()
    d=detector.detect_faces(img)
    d=d[0]['box']
    x,y,w,h=d
    img=img[y:y+h,x:x+w]
    #img=preprocess_input(cv2.resize(img,taget_size))
    img=cv2.resize(img,taget_size)
    img_array=np.expand_dims(img,axis=0)
    
    return img_array
     
    
from mtcnn import MTCNN
import numpy as np
import cv2
import matplotlib.pyplot as plt
#from keras_vggface.utils import preprocess_input

def extract_face(file_name,taget_size=(224,224)):
    img=cv2.imread(file_name)
    #img=img.astype('float32')
    detector=MTCNN()
    d=detector.detect_faces(img)
    d=d[0]['box']
    x,y,w,h=d
    img=img[y:y+h,x:x+w]
    #img=preprocess_input(cv2.resize(img,taget_size))
    
    #img_array=np.expand_dims(img,axis=0)
    
    
    
    return img
    

def cos_similarity(img_rep1,img_rep2):
  
  dot = np.dot(img_rep1, img_rep2.T)
  norma = np.linalg.norm(img_rep1)
  normb = np.linalg.norm(img_rep2)
  cos = dot / (norma * normb)
  return (cos)
  

threshold=0.5
def face_matching(img1,img2):
  img_rep1=get_model().predict(crop_image(img1))
  img_rep2=get_model().predict(crop_image(img2))
  cos_score=cos_similarity(img_rep1,img_rep2)
  #img1_arr=extract_face(img1)
  #img2_arr=extract_face(img2)
  #plt.subplot(2,1)
  #fig, (ax1, ax2) = plt.subplots(1, 2)
  #ax1.imshow(img1_arr)
  #ax2.imshow(img2_arr)

  if cos_score >=threshold:
    return True

  else:
    return False


import os
threshold=0.5
def face_matching_folder(img,folder_path):

  path_list=os.listdir(folder_path)
  img_rep=get_model().predict(crop_image(img))
   
  d={}
  for i in path_list:
    if i.endswith("jpg"):
        d[i]=cos_similarity(img_rep,get_model().predict(crop_image(folder_path+str('/')+i)))

  for i,j in d.items():
    if j>=0.5:
      #print("image processed is matching with %s"%i)
      return i
    
  return 'uknown'

    #else:
      #print("image processed is not matching with %s"%i)

    #else:
      #print("zero matching")
files = {'file1': '/home/deeplearningcv/Downloads/fraud_detection/images/omh.jpg','file2':'/home/deeplearningcv/Downloads/fraud_detection/images/oms.jpg'}     

#print(face_matching(files.get('file1'),files.get('file2')))




