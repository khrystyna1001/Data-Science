import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model

def main():
    DataDir = r"./chest_xray/train/"
    CATEGORIES = ["NORMAL", "PNEUMONIA"]

    # converting images into grayscale
    for i in CATEGORIES:
        path = os.path.join(DataDir, i)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            plt.imshow(img_array, cmap='gray')
            plt.show()
            break
        break
    
    # resize image
    img_size = 100
    new_array = cv2.resize(img_array, (img_size,img_size))
    plt.imshow(new_array, cmap='gray')
    plt.show()

    # resizing & grayscale
    training_data = []
    def create_training_data():
        for i in CATEGORIES:
            path = os.path.join(DataDir, i)
            class_num = CATEGORIES.index(i)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (img_size,img_size))
                    training_data.append([new_array,class_num])
                except Exception as e:
                    pass
    create_training_data()
    print(len(training_data))   

    # shuffle data
    import random
    random.shuffle(training_data)
    for sample in training_data[:10]:
        print(sample)

    X = []
    y = []
    
    # spliting the features and labels
    for features,label in training_data:
        X.append(features)
        y.append(label)

    print(X[0].reshape(-1,img_size,img_size,1))

    # reshapping the features for making it compatible with tensorflow
    X = np.array(X).reshape(-1,img_size,img_size,1)
    y = np.array(y)

    validation_data = []
    DataDir_val = r"./chest_xray/val/"

    def create_validating_data():
        for i in CATEGORIES:

            path=os.path.join(DataDir_val,i)
            class_num=CATEGORIES.index(i)

            for img in os.listdir(path):
                try:
                    img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                    new_array=cv2.resize(img_array,(img_size,img_size))
                    validation_data.append([new_array,class_num])

                except Exception as e:
                    pass
    
    create_validating_data()
    print(len(validation_data))

    # same for validation data
    import random
    random.shuffle(validation_data)
    for sample in validation_data[:10]:
        print(sample)

    X_val = []
    y_val = []
    for features,label in validation_data:
        X_val.append(features)
        y_val.append(label)
    y_val = np.array(y_val)
    X_val=np.array(X_val).reshape(-1,img_size,img_size,1)

    from keras.models import Sequential
    from keras.layers import Dense,Dropout,Activation,Flatten, Conv2D,MaxPooling2D
    import pickle

    # feature scaling
    X = X/255.0
    x_val = X_val/255.0

    # intializing the neural network layer for training the model
    # model= Sequential()

    # model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(.2))

    # model.add(Conv2D(128,(3,3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(.2))

    # model.add(Conv2D(256,(3,3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(.2))

    # model.add(Flatten())
    # model.add(Dense(64))

    # model.add(Dropout(.5))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))

    # # model summary
    # print(model.summary())

    # # compile model
    # model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    # model.fit(X,y,batch_size=4,epochs=10,validation_data=(x_val, y_val))

    # saving the model
    # model.save(r"model_10.h5")
    model = tf.keras.models.load_model("./chest_xray/model_10.h5")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def prepare(image):
        img_size=100
        #img_array=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
        img=tf.keras.preprocessing.image.load_img(image, color_mode='grayscale', target_size=(img_size,img_size))
        new_array=tf.keras.preprocessing.image.img_to_array(img)

        return new_array.reshape(-1,img_size,img_size,1)

    image = r'./chest_xray/test/NORMAL/IM-0059-0001.jpeg'
    prediction1=model.predict([prepare(image)/255.0])
    print(prediction1)
    print(round(prediction1[0][0]))
    print(CATEGORIES[int(round(prediction1[0][0]))])

    image2 = r'./chest_xray/test/PNEUMONIA/person1_virus_7.jpeg'
    prediction2=model.predict([prepare(image2)/255.0])
    print(prediction2)
    print(round(prediction2[0][0]))
    print(CATEGORIES[int(round(prediction2[0][0]))])

    img=mpimg.imread(image)
    imgplot=plt.imshow(img)
    plt.title(CATEGORIES[int(prediction1[0][0])])
    plt.show()

    img2=mpimg.imread(image2)
    imgplot2=plt.imshow(img2)
    plt.title(CATEGORIES[int(prediction2[0][0])])
    plt.show()
    
    from keras.models import load_model

    # set the path for test data
    test_dir = "./chest_xray/test"

    # initialize lists for storing test data
    X_test = []
    y_test = []

    # loop through the test data directory and extract the images and their labels
    for category in CATEGORIES:
        path = os.path.join(test_dir, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                X_test.append(new_array)
                y_test.append(class_num)
            except Exception as e:
                pass

    # convert test data to numpy arrays
    X_test = np.array(X_test).reshape(-1, img_size, img_size, 1)
    y_test = np.array(y_test)

    # normalize test data
    X_test = X_test / 255.0

    # calculate test accuracy
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    # print test accuracy
    print('Test accuracy:', test_acc)

    # Using Pre Trained Model
    from tensorflow.keras.applications.vgg16 import VGG16
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # load the pre-trained model (VGG16)
    base_model = VGG16(input_shape=(img_size,img_size,3), include_top=False, weights='imagenet')

    # freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False

    # add custom layers for classification
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # create a new model
    model = Model(inputs=base_model.input, outputs=predictions)

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # define the image generators for training and validation data
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)

    # specify the training and validation data directories
    train_dir = './chest_xray/train'
    val_dir = './chest_xray/val'

    # create the image generators for training and validation data
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_size, img_size), batch_size=32, class_mode='binary')
    val_generator = val_datagen.flow_from_directory(val_dir, target_size=(img_size, img_size), batch_size=32, class_mode='binary')

    # train the model
    model.fit(train_generator, epochs=10, validation_data=val_generator)

    # evaluate the model on test data
    test_dir = './chest_xray/test'
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_size, img_size), batch_size=32, class_mode='binary', shuffle=False)
    test_loss, test_acc = model.evaluate(test_generator)
    print('Test accuracy:', test_acc)

    model = tf.keras.models.load_model("./chest_xray/custom_pre_trained_model_10.h5")
    
    # model summary
    print(model.summary())

if __name__ == "__main__":
    main()