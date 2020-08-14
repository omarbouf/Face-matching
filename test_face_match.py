
import requests
import json

def test_face_match():
    url = 'http://127.0.0.1:5000/upl'
    # open file in binary mode
    files = {
    'file1' :open("/home/deeplearningcv/Downloads/oms.jpg",'rb'),
    'file2':open('/home/deeplearningcv/Downloads/omh.jpg','rb')}


    resp = requests.post(url, files=files)
    print(resp.text)
    #print( 'face_match response:\n', json.dumps(resp.json()) )
    
if __name__ == '__main__':
    test_face_match()