import os
import ray
import keras
import time
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# Cluster configuration
cluster_spec = {
    'worker': [
        '172.16.36.82:12345',
        '172.16.36.81:12345'
    ]
}

# Set TF_CONFIG environment variable
# export WORKER_INDEX=0      ====> FOR MASTER GPU02
# export WORKER_INDEX=1      ====> FOR SLAVE GPU01
# This file sould be executed on both GPUs

tf_config = {
    'cluster': cluster_spec,
    'task': {'type': 'worker', 'index': int(os.environ['WORKER_INDEX'])}  # WORKER_INDEX should be set for each worker
}
os.environ['TF_CONFIG'] = json.dumps(tf_config)


# Step 2: Define the Multi-Worker Strategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Step 3: Setup the Distributed Training
with strategy.scope():
    img_height, img_width = 224, 224
    channel = 3
    num_classes = 3
    input_shape=(None,img_height,img_width, channel)
    batch_size = 32
    #train_ds = tf.keras.utils.image_dataset_from_directory("/home/gashugi/Th/fruits/train", image_size = (img_height, img_width), batch_size = batch_size)
    #val_ds = tf.keras.utils.image_dataset_from_directory("/home/gashugi/Th/fruits/validation", image_size = (img_height, img_width), batch_size = batch_size) 
    #test_ds = tf.keras.utils.image_dataset_from_directory("/home/gashugi/Th/fruits/test", image_size = (img_height, img_width), batch_size = batch_size)

    # Data preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        "Beans_diseases_datasetT2/train",
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        "Beans_diseases_datasetT2/validation",
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    # Load the saved MobileNetV2 model
    base_model = tf.keras.models.load_model('mobilenet_v2.h5')

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    # Add classification head
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    #model.summary()
    #model.compile(optimizer="adam", loss=tf.losses.SparseCategoricalCrossentropy(from_logits = True), metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # Step 5: Train model

    start_time = time.time()
    #history = model.fit(train_ds, validation_data = val_ds, epochs = 64)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples/train_generator.batch_size,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples/validation_generator.batch_size)

    training_time = time.time() - start_time
    print("Traininig Time: ", training_time)
    
    # Step 6: Evaluate the Model
    test_generator = test_datagen.flow_from_directory(
        "Beans_diseases_datasetT2/test",
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    y_true = test_generator.classes
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("precision:", precision)
    print("Traininig Time: ", training_time)

    # Print confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=train_generator.class_indices.keys()))

    #Save the model
    #Model conversion to tflite in order to be compatible with Android Phone

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open("model1006.tflite", 'wb') as f:
        f.write(tflite_model)














