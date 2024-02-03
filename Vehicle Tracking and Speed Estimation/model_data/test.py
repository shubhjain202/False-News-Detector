import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 256

test_datagen = ImageDataGenerator(rescale = 1/255)

test_generator = test_datagen.flow_from_directory(
    'crops_test',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    class_mode="sparse"
)

# Evaluate the model on the test dataset
model = tf.keras.models.load_model('vehicle_train')

print(model.summary())
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
results = model.evaluate(test_generator)
print("Test loss, Test accuracy:", results)
model.save('best_model_for_test.h5')
