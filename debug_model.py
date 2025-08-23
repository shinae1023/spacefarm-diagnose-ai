# debug_model.py
import tensorflow as tf
import numpy as np
import os

def debug_model():
    model_path = 'cabbage_disease_model.keras'
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Model exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        print("❌ 모델 파일이 없습니다!")
        return False
    
    try:
        print("🔄 모델 로딩 시도...")
        model = tf.keras.models.load_model(model_path)
        print("✅ 모델 로딩 성공!")
        
        # 모델 구조 확인
        print("\n📋 모델 요약:")
        model.summary()
        
        # 입력 형태 확인
        print(f"\n📏 입력 형태: {model.input_shape}")
        print(f"출력 형태: {model.output_shape}")
        
        # 테스트 예측
        print("\n🧪 테스트 예측...")
        test_input = np.random.random((1, 150, 150, 3)).astype(np.float32)
        print(f"테스트 입력 형태: {test_input.shape}")
        
        prediction = model.predict(test_input, verbose=0)
        print(f"✅ 예측 성공! 결과: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        print(f"에러 타입: {type(e).__name__}")
        
        # 상세 에러 정보
        import traceback
        print("\n상세 에러:")
        traceback.print_exc()
        
        return False

def create_simple_model():
    """간단한 테스트 모델 생성"""
    print("\n🔨 간단한 테스트 모델 생성...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # 테스트 모델 저장
    model.save('test_model.keras')
    print("✅ 테스트 모델 저장 완료: test_model.keras")
    
    # 테스트 예측
    test_input = np.random.random((1, 150, 150, 3)).astype(np.float32)
    prediction = model.predict(test_input, verbose=0)
    print(f"✅ 테스트 모델 예측 성공: {prediction}")

if __name__ == "__main__":
    print("🔍 모델 디버깅 시작...\n")
    
    success = debug_model()
    
    if not success:
        print("\n❌ 기존 모델에 문제가 있습니다.")
        print("🔨 새로운 테스트 모델을 생성합니다...")
        create_simple_model()
        print("\n💡 test_model.keras 파일을 cabbage_disease_model.keras로 이름을 바꾸고 다시 배포해보세요.")
    else:
        print("\n✅ 모델에는 문제가 없습니다. 다른 원인을 찾아보겠습니다.")