from flask import Flask, request, jsonify
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import io
import os

app = Flask(__name__)

# .tflite 모델 로드
try:
    # `flask_app.py` 파일과 `cabbage_disease_model.tflite` 파일이 같은 폴더에 있어야 합니다.
    model_path = os.path.join(os.path.dirname(__file__), 'cabbage_disease_model.tflite')
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # 첫 요청 지연을 줄이기 위해 미리 모델 호출
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], np.zeros(input_shape, dtype=np.float32))
    interpreter.invoke()
    
except Exception as e:
    interpreter = None
    print(f"Error loading TFLite model: {e}")

@app.route('/diagnose', methods=['POST'])
def diagnose_cabbage():
    if interpreter is None:
        return jsonify({"error": "Model not loaded"}), 500

    # 파일 확인
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 이미지 전처리 및 예측
    try:
        # 이미지 크기 조정
        image = Image.open(io.BytesIO(file.read())).resize((150, 150))
        # 이미지 데이터를 float32 타입으로 변경하고 정규화
        image = np.array(image, dtype=np.float32) / 255.0
        # 배치 차원 추가
        image = np.expand_dims(image, axis=0)
        
        # TFLite 모델 예측
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        result = "disease" if prediction > 0.5 else "normal"
        
        return jsonify({"result": result, "confidence": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500