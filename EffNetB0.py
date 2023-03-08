import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

folder = 'flowers'

labels = ['sunflower', 'daisy', 'tulip', 'dandelion', 'rose']
all_pics = []
# image_folders = [os.path.join(folder, f) for f in os.listdir(folder)]
for label in labels:
    print(label)
    class_num = labels.index(label)
    im_folder = os.path.join(folder, label)
    # print(im_folder)
    # print(os.listdir(im_folder))
    for img in os.listdir(im_folder):
        img_path = os.path.join(im_folder, img)
        with Image.open(img_path) as flower:
            flower = flower.resize((224, 224))
            all_pics.append([np.asarray(flower), class_num])


X, y = [], []
for flower, num_label in all_pics:
    X.append(flower)
    y.append(num_label)

X = np.array(X, dtype=np.float32)
X = X/255
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=228)

X_train = X_train[:1000]
y_train = y_train[:1000]
X_test = X_test[:200]
y_test = y_test[:200]

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

# tf.keras.backend.clear_session()
# Define the EfficientNet model
base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights=None, classes=5)

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(5, activation='softmax')
])
# model.summary()
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # добавить learning_rate

start_time = datetime.now()
# Train the model on your image and label arrays
history = model.fit(x=X_train, y=y_train, validation_split=0.2, epochs=100)

print('fitting time:', datetime.now() - start_time)
pd.DataFrame(history.history).plot()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim([0, 4])
plt.show()

model.evaluate(X_test, y_test)
predictions = model.predict(X_test)
results = np.argmax(predictions, axis=1)
print('y_test[:20]', y_test[:20])
print('results[:20]', results[:20])

fig, ax = plt.subplots(4, 2)
fig.set_size_inches(15,15)
n = 0
for i in range(4):
  for j in range(2):
    ax[i,j].imshow(X_test[n])
    ax[i,j].set_title('Predicted flower: ' + labels[results[n]] + '\nActual flower: ' + labels[y_test[n]])
    n += 1
plt.tight_layout()
plt.show()