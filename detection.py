from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# YOLO 모델 불러오기
net = cv2.dnn.readNet("C:/yolopj/yolov4.weights", "C:/yolopj/yolov4.cfg")

# COCO 데이터셋의 클래스 이름 불러오기
with open("C:/yolopj/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# YOLO 네트워크의 레이어 이름들 가져오기
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

@app.route('/detect', methods=['POST'])
def detect_objects():
    file = request.files['image'].read()  # 이미지를 받습니다.
    npimg = np.frombuffer(file, np.uint8)  # numpy.fromstring()은 deprecated 되었기 때문에 frombuffer()를 사용합니다.
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    detections = net.forward(output_layers)

    result = []

    for detection in detections:
        for object_detection in detection:
            scores = object_detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                center_x, center_y, width, height = (object_detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype('int')
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                result.append({
                    'label': classes[class_id],
                    'confidence': float(confidence),
                    'x': x, 'y': y, 'width': width, 'height': height
                })

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
