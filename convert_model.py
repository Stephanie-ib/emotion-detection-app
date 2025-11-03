import tensorflow as tf
from tensorflow import keras

print(f"🔍 Your TensorFlow version: {tf.__version__}")
print("🔄 Loading model trained with TensorFlow 2.20.0...")

try:
    model = keras.models.load_model('face_emotionModel.h5')
    print("✅ Model loaded successfully!")
    
    print("💾 Re-saving without optimizer state...")
    model.save('face_emotionModel.h5', save_format='h5', include_optimizer=False)
    print("✅ Model converted and saved!")
    print("🎉 This model will now work with TensorFlow 2.13.0 on Render!")
    
except Exception as e:
    print(f"❌ Error: {e}")
