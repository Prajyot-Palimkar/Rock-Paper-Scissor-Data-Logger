
# Rock Paper Scissor Data Logger

The motivation behind making this application is to play rock paper scissor in the isolation ward. It is a data logger that counts the number of times each of the three signal is recognized.

## Installing Libraries

To run this application, you just need to import the few libraries. Following is the code to import the required libraries.

```bash
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense,MaxPool2D,Dropout,Flatten,Conv2D,GlobalAveragePooling2D,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from random import choice,shuffle
from scipy import stats as st
from collections import deque
```


## Load the model

Once the required libraries are installed and imported you are ready to go. Now you need train the model. For your easy go I had already trained the model and saved it. You can find the trained model [rps4.h5](https://drive.google.com/file/d/1FBHmCptx-l4-xOLolAa8LAJpT1HCwt75/view?usp=sharing) from here. One can load my model by following code.


```bash
model = load_model("rps4.h5")
```


## Data Logging

After the model is loaded, we need to run the following code for the data logger. Using this code, you can log the number of times each of the three signal was shown and save it to a database.

```bash
cap = cv2.VideoCapture(0)
box_size = 234
width = int(cap.get(3))

# Specify the number of attempts done intially, i.e. 5.
rock_attempts = 0
paper_attempts = 0
scissor_attempts = 0

# Initially the moves will be 'nothing'
final_user_move = "nothing"

label_names = ['nothing', 'paper', 'rock', 'scissor']

# The default color of bounding box is Blue
rect_color = (255, 0, 0)

# This variable remembers if the hand is inside the box or not.
hand_inside = False

# At each iteration we will increase the total_attempts of each signal value by 1
rock_total_attempts = rock_attempts
paper_total_attempts = paper_attempts
scissor_total_attempts = scissor_attempts

# We will only consider predictions having confidence above this threshold.
confidence_threshold = 0.70

# Instead of working on a single prediction, we will take the mode of 5 predictions by using a deque object
# This way even if we face a false positive, we would easily ignore it
smooth_factor = 5

# Our initial deque list will have 'nothing' repeated 5 times.
de = deque(['nothing'] * 5, maxlen=smooth_factor)

while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
           
    cv2.namedWindow("Rock Paper Scissors", cv2.WINDOW_NORMAL)

    # extract the region of image within the user rectangle
    roi = frame[5: box_size-5 , width-box_size + 5: width -5]
    
    roi = np.array([roi]).astype('float64') / 255.0
    
    # Predict the move made
    pred = model.predict(roi)
    
    # Get the index of the predicted class
    move_code = np.argmax(pred[0])
   
    # Get the class name of the predicted class
    user_move = label_names[move_code]
    
    # Get the confidence of the predicted class
    prob = np.max(pred[0])
    
    # Make sure the probability is above our defined threshold
    if prob >= confidence_threshold:
        
        # Now add the move to deque list from left
        de.appendleft(user_move)
        
        # Get the mode i.e. which class has occured more frequently in the last 5 moves.
        try:
            final_user_move = st.mode(de)[0][0] 
            
        except StatisticsError:
            print('Stats error')
            continue
             
        # If nothing is not true and hand_inside is False then proceed.
        # Basically the hand_inside variable is helping us to not repeatedly predict during the loop
        # So now the user has to take his hands out of the box for every new prediction.
        
        if final_user_move != "nothing" and hand_inside == False:
            
            # Set hand inside to True
            hand_inside = True 
            
            # Add one attempt
            if final_user_move == "rock":
                rock_total_attempts += 1
            elif final_user_move == "paper":
                paper_total_attempts += 1
            elif final_user_move == "scissor":
                scissor_total_attempts += 1
            
        # If class is nothing then hand_inside becomes False
        elif final_user_move == 'nothing':            
            hand_inside = False
            rect_color = (255, 0, 0) 

    # This is where all annotation is happening. 

    cv2.putText(frame, "Your Move: " + final_user_move,
                    (190, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, "Rock Attempts done: {}".format(rock_total_attempts),
                    (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Paper Attempts done: {}".format(paper_total_attempts),
                    (2, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 2, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Scissor Attempts done: {}".format(scissor_total_attempts),
                (190, 400), cv2.FONT_HERSHEY_COMPLEX, 0.7,(100, 2, 255), 1, cv2.LINE_AA)    
    
    cv2.rectangle(frame, (width - box_size, 0), (width, box_size), rect_color, 2)
    
    with open("rockpaperscissor.csv",'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Rock Attempt","Paper Attempt","Scissor Attempt"])
        writer.writerow([rock_total_attempts,paper_total_attempts,scissor_total_attempts])

    # Display the image    
    cv2.imshow("Rock Paper Scissors", frame)

    # Exit if 'q' is pressed 
    k = cv2.waitKey(10)
    if k == ord('q'):
        break

# Relase the camera and destroy all windows.
cap.release()
cv2.destroyAllWindows()
```
