import tensorflow as tf

def convert_model():
    try:
        # 모델 로드 (compile=False 옵션 사용)
        model = tf.keras.models.load_model('cabbage_disease_model.keras', compile=False)

        # TFLiteConverter 설정
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # 모델 최적화 (양자화 비활성화)
        # 32비트 부동 소수점(float32)을 사용하여 정밀도 유지
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, # TFLite 기본 연산
            tf.lite.OpsSet.SELECT_TF_OPS # TensorFlow 연산
        ]
        converter.target_spec.supported_types = [tf.float32] # float32 명시

        # 모델 변환
        tflite_model = converter.convert()

        # 파일 저장
        with open('modelcabbage_disease_model.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print("Model conversion successful!")

    except Exception as e:
        print(f"An error occurred during conversion:\n{e}")

if __name__ == "__main__":
    convert_model()