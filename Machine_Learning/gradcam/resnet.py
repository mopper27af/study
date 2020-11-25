from keras.preprocessing import image
from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense

from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import pandas as pd

from sklearn.metrics import accuracy_score

def get_model(nb_classes,img_width=128, img_height=128):
    input_tensor = Input(shape=(img_width, img_height, 3))
    # include_top=False 出力層なし
    model = ResNet50(include_top=False,classes=nb_classes, weights='imagenet',input_tensor=input_tensor)
    #  model = VGG16(include_top=False,classes=nb_classes, weights='imagenet',input_tensor=input_tensor)
    x=model.output
    x = Flatten()(x)
    # x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation='softmax')(x)
    # VGG16とFC層を結合してモデルを作成（完成図が上の図）
    model= Model(input=model.input, output=x)

    # VGG16の図の青色の部分は重みを固定（frozen）
    for layer in model.layers[:15]:
        layer.trainable = False

    # 多クラス分類を指定
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
	img_width=64
	img_height=64
	is_learning=True

	resnet_model = get_model(2,img_width,img_height)
	# トレーンング用、バリデーション用データを生成するジェネレータ作成
	train_datagen = ImageDataGenerator(
	  rescale=1.0 / 255,
	  #すでに画像の水増し済みの方は、下記２行は必要ありません。
	  #zoom_range=0.2,
	  #horizontal_flip=True
	)

	validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

	# 訓練データ保存先
	# /保存先フォルダ/ {0 or 1 } / *.png
	classes=["0","1"]
	classes_name =["輿水幸子","森久保乃々"]
	train_data_dir="Y:/MLearningData/SatiMori/"
	#トレーニングデータ用の画像数
	#バッチサイズ
	batch_size = 2


	# 訓練データ
	train_generator = train_datagen.flow_from_directory(
	  train_data_dir,
	  target_size=(img_width, img_height),
	  color_mode='rgb',
	  classes=classes,
	  class_mode='categorical',
	  batch_size=batch_size,
	  shuffle=True)


	# テストデータ
	validation_data_dir=  "Y:/MLearningData/SatiMori/test"
	validation_generator = validation_datagen.flow_from_directory(
	  validation_data_dir,
	  target_size=(img_width, img_height),
	  color_mode='rgb',
	  classes=classes,
	  class_mode='categorical',
	  batch_size=batch_size,
	  shuffle=True)
	  
	# テストデータ（評価用
	validation_data_dir=  "Y:/MLearningData/SatiMori/test"
	validation_generator2 = validation_datagen.flow_from_directory(
	  validation_data_dir,
	  target_size=(img_width, img_height),
	  color_mode='rgb',
	  classes=classes,
	  class_mode='categorical',
	  batch_size=1,
	  shuffle=False)
	  
	nb_epoch=1
	print("learning start")
	# 訓練する場合
	if is_learning:
	    # Fine-tuning
	    history = resnet_model.fit_generator(
	        train_generator,
	        # samples_per_epoch=nb_train_samples,
	        nb_epoch=nb_epoch,
	        validation_data=validation_generator,
	        # nb_val_samples=nb_validation_samples
	    )
	    resnet_model.save('res_model') 
	    resnet_model.save_weights("res_model_weights")
	else:
	    resnet_model.load_weights("res_model_weights")
	
	print("learning end")
  