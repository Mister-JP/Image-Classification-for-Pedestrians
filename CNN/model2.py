from matplotlib import pyplot as plt
from Start import *
import keras
import tensorflow as tf
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

path = "C:\\VT\\fall21\\computer vision\\diamler dataset\\DaimlerBenchmark\\Data\\TrainingData"

batch_size = 64
epochs = 20
num_classes = 2

pos = path+"\\samplepos"
neg = path + "\\sampleneg"

train_x,valid_x,train_label,valid_label = dataprep(pos,neg)

print(" In model train_y = ", train_label.shape)
print(" In model type train_y = ", type(train_label))
print("In model train y -",train_label)
print("In model train y -",train_x)


fashion_model = Sequential()
fashion_model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same',input_shape=(96,48,1)))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.35))
fashion_model.add(Flatten())
fashion_model.add(Dense(256, activation='relu'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dropout(0.25))
fashion_model.add(Dense(num_classes, activation='softmax'))

print(fashion_model.summary())
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
fashion_train_dropout = fashion_model.fit(train_x, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_x, valid_label))
fashion_model.save("fashion_model_dropout_model2.h5py")
test_eval = fashion_model.evaluate(valid_x, valid_label, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

accuracy = fashion_train_dropout.history['accuracy']
val_accuracy = fashion_train_dropout.history['val_accuracy']
loss = fashion_train_dropout.history['loss']
val_loss = fashion_train_dropout.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

test_eval = fashion_model.evaluate(valid_x, valid_label, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

pred = fashion_model.predict(valid_x)
for i in range(len(valid_label)):
    print("real ans - ",valid_label[i]," pred ans - ", pred[i])

train_x=[]
pathpos = "C:\\VT\\fall21\\computer vision\\diamler dataset\\DaimlerBenchmark\\Data\\TrainingData\\Pedestrians\\48x96\\pos10355.pgm"
image = cv2.imread(pathpos,cv2.IMREAD_GRAYSCALE)
image = image.reshape(image.shape[0],image.shape[1],1)
train_x.append(image)
pathpos = "C:\\VT\\fall21\\computer vision\\diamler dataset\\DaimlerBenchmark\\Data\\TrainingData\\extractednp\\frame9878.pgm"
image = cv2.imread(pathpos,cv2.IMREAD_GRAYSCALE)
image = image.reshape(image.shape[0],image.shape[1],1)
train_x.append(image)
train_x = np.array(train_x)
pred = fashion_model.predict(valid_x)
print("pos - ",pred[0])
print("neg - ",pred[1])