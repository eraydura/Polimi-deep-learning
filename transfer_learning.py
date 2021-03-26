import json
import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.applications.vgg19 import VGG19,preprocess_input
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint,TerminateOnNaN
from tensorflow.python.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

FAST_RUN = False
IMAGE_WIDTH=224
IMAGE_HEIGHT=224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

#load json data
f = open('MaskDataset/train_gt.json')
data = json.load(f)

#divide file_name into categories
categories = []
file_name = []
for i in data:
    category = data[i]
    if category == 0:
        file_name.append(i)
        categories.append(0)
    elif category == 2:
        file_name.append(i)
        categories.append(2)
    else:
        file_name.append(i)
        categories.append(1)

# Closing file
f.close()

#create a dataframe including file_name and category together
df = pd.DataFrame({
    'filename': file_name,
    'category': categories
})

# VGG19 initialization
vgg = VGG19(weights="imagenet", include_top=False,input_tensor=Input(shape=(IMAGE_WIDTH, IMAGE_WIDTH,IMAGE_CHANNELS)))
for layer in vgg.layers:
     layer.trainable = False
model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(units=512, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dense(units=512, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=3, activation='softmax')) # 3 because we have mask, no mask and some mask

# optimizer as Adam with learning rate
opt = Adam(learning_rate=1e-4)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()

# Early Stopping
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

# Checkpoints between the training steps
model_checkpoint = ModelCheckpoint(filepath='VGG_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=20)

# Termination of training if the loss become Nan
terminate_on_nan = TerminateOnNaN()

# For watching the live loss, accuracy and graphs using tensorboard
t_board = TensorBoard(log_dir='./logs', histogram_freq=0,
                      batch_size=32, write_graph=True,
                      write_grads=False,write_images=False,
                      embeddings_freq=0, update_freq='epoch')

# For reducing the loss when loss hits a plateau. This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3,verbose=1,factor=0.5,  min_lr=0.000001)

callbacks = [earlystop, learning_rate_reduction,t_board,t_board,terminate_on_nan]

df["category"] = df["category"].replace({0: 'NO PERSON', 1: 'ALL THE PEOPLE', 2: 'SOME'})

# Split randomly 0.20 of train images as validation images
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15

# Create train ImageDataGenerator objects with data augmentation
train_data_gen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=10,
                                    width_shift_range=10,
                                    height_shift_range=10,
                                    zoom_range=0.3,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode='constant',
                                    cval=0,
                                    preprocessing_function=preprocess_input)


# Create validation ImageDataGenerator objects
valid_data_gen = ImageDataGenerator(rescale=1./255,preprocessing_function=preprocess_input)

train_generator = train_data_gen.flow_from_dataframe(
    train_df,
    "MaskDataset/training",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size,
)


validation_generator = valid_data_gen.flow_from_dataframe(
    validate_df,
    "MaskDataset/training",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

# after define a model, taking validation and train datas, we should fit model
epochs=3 if FAST_RUN else 50
history = model.fit_generator(
train_generator,
epochs = epochs,
validation_data = validation_generator,
validation_steps = total_validate // batch_size,
steps_per_epoch = total_train // batch_size,
callbacks = callbacks
)

#saving model
model.save("model.h5")

#create result plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))


legend= plt.legend(loc='best',shadow=True)
plt.tight_layout()
plt.show()


test_filenames = os.listdir("MaskDataset/test")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

# Create test ImageDataGenerator objects
test_gen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    "MaskDataset/test",
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)

#Predict test images
predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

#divide 3 categories into the prediction
test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({0: 'NO PERSON', 1: 'ALL THE PEOPLE', 2: 'SOME'})

#saving result in csv file
submission_df=test_df.copy()
submission_df['Id'] = submission_df['filename']
submission_df['Category'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)
