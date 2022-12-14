from flask import Flask, render_template, request
import cv2
import numpy as np
import base64

import backend.opencv

app = Flask(__name__)

@app.route("/")
def start_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image'].read()
    if len(bytearray(file)) == 0:
            return render_template('index.html', faceDetected=False, num_faces=0, init=True)
 
    # Read image
    image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_UNCHANGED)
    
    # Detect faces
    faces = backend.opencv.detect_faces(image)

    if len(faces) == 0:
        faceDetected = False
        num_faces = 0
        to_send = ''
    else:
        faceDetected = True
        num_faces = len(faces)
        
        # Draw a rectangle
        for item in faces:
            backend.opencv.draw_rectangle(image, item['rect'])
        
        # In memory
        image_content = cv2.imencode('.jpg', image)[1].tobytes()
        encoded_image = base64.encodebytes(image_content)
        to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')

    return render_template('index.html', faceDetected=faceDetected, num_faces=num_faces, image_to_show=to_send, init=True)

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=8080)