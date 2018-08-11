import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 150, 150
model_path = './models1/model.h5'
model_weights_path = './models1/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("Label: Pizza")
  elif answer == 1:
    print("Labels: Poodle")
  elif answer == 2:
    print("Label: Rose")

  return answer

car_t = 0
car_f = 0
lab_t = 0
lab_f = 0
motocycle_t = 0
motocycle_f = 0
plane_t=0
plane_f=0
ship_t=0
ship_f=0
toktok_t=0
toktok_f=0
train_t=0
train_f=0
truck_t=0
truck_f=0

for i, ret in enumerate(os.walk('./test-data/car')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: car")
    result = predict(ret[0] + '/' + filename)
    if result == 0:
      car_t += 1
    else:
      car_f += 1

for i, ret in enumerate(os.walk('./test-data/lab')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: lab")
    result = predict(ret[0] + '/' + filename)
    if result == 1:
      lab_t += 1
    else:
      lab_f += 1

for i, ret in enumerate(os.walk('./test-data/motocycle')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: motocycle")
    result = predict(ret[0] + '/' + filename)
    if result == 2:
      motocycle_t += 1
    else:
      motocycle_f += 1
for i, ret in enumerate(os.walk('./test-data/plane')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    print("Label: plane")
    result = predict(ret[0] + '/' + filename)
    if result == 3:
      plane_t += 1
    else:
      plane_f += 1
 

for i, ret in enumerate(os.walk('./test-data/ship')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
        continue
    print("Label: ship")
    result = predict(ret[0] + '/' + filename)
    if result == 4:
        ship_t += 1
    else:
        ship_f += 1
        
        
for i, ret in enumerate(os.walk('./test-data/toktok')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
        continue
    print("Label: toktok")
    result = predict(ret[0] + '/' + filename)
    if result == 5:
       toktok_t += 1
    else:
       toktok_f += 1

        
for i, ret in enumerate(os.walk('./test-data/train')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
        continue
    print("Label: train")
    result = predict(ret[0] + '/' + filename)
    if result == 6:
       train_t += 1
    else:
       train_f += 1
       
for i, ret in enumerate(os.walk('./test-data/truck')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
        continue
    print("Label: train")
    result = predict(ret[0] + '/' + filename)
    if result == 7:
      truck_t += 1
    else:
       truck_f += 1
 

"""
Check metrics
"""
print("True car: ", car_t)
print("False car: ", car_f)
print("True lab: ", lab_t)
print("False lab: ", lab_f)
print("True motocycle ", motocycle_t)
print("False motocycle: ", motocycle_f)
print("True plane: ", plane_t)
print("False plane: ", plane_f)
print("True ship: ", ship_t)
print("False ship: ", ship_f)
print("True toktok: ", toktok_t)
print("False toktok: ", toktok_f)
print("True train: ", train_t)
print("False train: ", train_f)
print("True truck: ", truck_t)
print("False truck: ", truck_f)
