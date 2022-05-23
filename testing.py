# SOURCE :
# https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
# https://github.com/nicknochnack/YOLO-Drowsiness-Detection/blob/main/Drowsiness%20Detection%20Tutorial.ipynb
# https://github.com/tzutalin/labelImg

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Install and Import Dependencies
import numpy as np
import cv2

import time
from math import sqrt
from matplotlib import pyplot as plt



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Global Variable and Label

# Global Variable
frames = 0
o_move = 0
starttime = time.time()

wrist = [0,0]       #wrist = [None,None]

d_wrist_comp = 0
d_timer_comp = 0

d_wrist = 0
d_attention = 0
d_attention_comp = 0

d_attention_list = [0]
d_diagonal_list = [0]
str_object_focus = "unknown object"

hand_status = False
first_data = True

speed_eye = 0
speed_hand = 0

object_temp = ""
objects_name_all = []
objects_name_saved = []

objects_inter_all = []
objects_inter_string = []
objects_grasp_all = []

time_string = []
time_interact_string = []
location_string = []
location_interact_string = []

# Label
group_id = 0
surface = 'blank'
measurement_id = 0
series_id = 0         


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> STORE DATA TO CSV 

# Start Time
from asyncio.windows_events import NULL
import time
from datetime import datetime
start_time = time.time()

# Create CSV file
import csv

header_pos = [ 'timer', 
            'row_id', 'series_id', 'measurement_id', 'group_id', 'surface', 
            #'o_move', 'o_length', 'o_width', 
            'd_tip_0','d_tip_1', 'd_tip_2', 'd_tip_3', 'd_tip_4', 
            'd_pinch_1', 'd_pinch_2', 'd_pinch_3', 'd_pinch_4']      

csvfile_pos = open('hand_position.csv', 'w')
writer_pos = csv.writer(csvfile_pos, delimiter = ',', lineterminator='\n')
writer_pos.writerow(header_pos)

header_ang = [ 'timer', 
            'angle01', 'angle02', 'angle03', 
            'angle11', 'angle12', 'angle13',
            'angle21', 'angle22', 'angle23', 
            'angle31', 'angle32', 'angle33',
            'angle41', 'angle42', 'angle43',] 
        
csvfile_ang = open('hand_angle.csv', 'w')
writer_ang = csv.writer(csvfile_ang, delimiter = ',', lineterminator='\n')
writer_ang.writerow(header_ang)

header_eye = [ 'timer', 'wrist_x', 'wrist_y','eye_x', 'eye_y'] 
csvfile_eye = open('hand_eye.csv', 'w')
writer_eye = csv.writer(csvfile_eye, delimiter = ',', lineterminator='\n')
writer_eye.writerow(header_eye)

header_speed = [ 'timer', 'd_attention', 'd_wrist', 'speed_eye', 'speed_wrist'] 
csvfile_speed = open('hand_speed.csv', 'w')
writer_speed = csv.writer(csvfile_speed, delimiter = ',', lineterminator='\n')
writer_speed.writerow(header_speed)



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  SOCKET UDP 

import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5005



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  DATE TIME

from abc import abstractmethod
import datetime
from datetime import date
from datetime import datetime

from numba.core.types.misc import Object

# Get time
now = datetime.now().time()
current_time = now.strftime("%H:%M:%S")

toc_0 = now.strftime("%H:%M:%S")
toc_1 = now.strftime("%H:%M:%S")
toc_2 = now.strftime("%H:%M:%S")
toc_3 = now.strftime("%H:%M:%S")
toc_4 = now.strftime("%H:%M:%S")





#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  GRAPH DATABASE (Neo4j)

from neo4j import GraphDatabase
from pandas import DataFrame
import itertools

class Neo4jConnection:
    
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)
    
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
    
    def query(self, query, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            response = list(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response

conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", pwd="mobi")
conn.query("MATCH (a) -[r] -> () DELETE a,r")
conn.query("MATCH (a) DELETE a")
#conn.query("CREATE OR REPLACE DATABASE neo4j")

# Create Node:Person
query_string = 'CREATE (person:Person {name:"person"})'
conn.query(query_string, db='neo4j')





#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SETUP NEURAL NETWORKS LSTM

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # -> x needs to be: (batch_size, seq, input_size)

        #self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)            # <<<<<<<<<<< RNN
        # or:
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)           # <<<<<<<<<<< GRU
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)         # <<<<<<<<<<< LSTM
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)          # <<<<<<<<<<< LSTM
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        #out, _ = self.rnn(x, h0)                                                            # <<<<<<<<<<< RNN, GRU
        # or:
        out, _ = self.lstm(x, (h0,c0))                                                       # <<<<<<<<<<< LSTM
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        return out

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_classes = 4
num_epochs = 100
batch_size = 1
learning_rate = 0.001

input_size = 9
sequence_data = 10
hidden_size = 128
num_layers = 2

# Defining ANN Architechture
model_nn = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
model_nn.load_state_dict(torch.load(r"C:\Users\Azhar Aulia Saputra\MY_CODE\PyTorch for Action Recognition\model_gru.pkl"))
model_nn.to(device)
model_nn.eval()

import collections
coll_hand = collections.deque(maxlen=sequence_data)

import pickle
sc_input = pickle.load(open(r"C:\Users\Azhar Aulia Saputra\MY_CODE\PyTorch for Action Recognition\scaler_input.pkl",'rb'))





#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HAND TRACKING : MEDIAPIPE

### HAND TRACKING: SETUP

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

joint_list_0 = [[2,1,0], [3,2,1], [4,3,2]]
joint_list_1 = [[6,5,0], [7,6,5], [8,7,6]]
joint_list_2 = [[10,9,0], [11,10,9], [12,11,10]]
joint_list_3 = [[14,13,0], [15,14,13], [16,15,14]]
joint_list_4 = [[18,17,0], [19,18,17], [20,19,18]]

### HAND TRACKING: FUCTION 

def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640,480]).astype(int))
            
            output = text, coords
            
    return output


def draw_finger_angles(image, results, joint_list):
    
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle
                
            cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [1920, 1080]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
    return image


def get_finger_angles(results, joint_list):
    
    finger_angles=[]

    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        joint_no = 1
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord

            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle
            
            if joint_no == 1 and angle < 90 :
                angle = 90
            elif joint_no == 2 and angle < 110 :
                angle = 110
            elif joint_no == 3 and angle < 90 :
                angle = 90
            
            joint_no = joint_no + 1
            finger_angles.append(round(angle, 2))

    return finger_angles





#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Setup OpenCV

PATH = r"C:\Users\Azhar Aulia Saputra\Desktop\cup02.mp4"    #  hand02.mp4
#PATH = 'rtsp://192.168.75.51:8554/live/all'                # <<<<<<<<<<<<<< From Tobii Pro Glasses

cap = cv2.VideoCapture(PATH) #0

ret,frame=cap.read()
vheight = frame.shape[0]
vwidth = frame.shape[1]
print ("Video size", vwidth,vheight)

#cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Stream', (960,540))


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Load Model

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Looping

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break
        frames += 1

        cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Stream', (960,540))
        
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Visual Attention Detector

        mask_eye = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        red_lower = np.array([0,175,125])
        red_upper = np.array([5,255,255])
        mask_eye = cv2.inRange(mask_eye, red_lower , red_upper)

        kernel = np.ones((3,3), np.uint8)
        mask_eye = cv2.dilate(mask_eye, kernel, iterations=20)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(mask_eye, cv2.MORPH_OPEN, kernel, iterations=1)
        close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

        cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        offset = 0

        red_ROI_number = 0

        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            #cv2.rectangle(frame, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 0, 255), 3)
            red_ROI_number += 1
            red_x = x + int(w/2)
            red_y = y + int(h/2)

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Hand Detection

        # Brightness and Contrast
        alpha = 2
        beta = 5
        frame = cv2.addWeighted(frame, alpha, np.zeros(frame.shape, frame.dtype), 0, beta)
        
        # BGR 2 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Set flag
        frame.flags.writeable = False
        # Hand Detections
        results = hands.process(frame)
        # Set flag to true
        frame.flags.writeable = True
        # RGB 2 BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        hand_angle = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        hand_position = [0,0,0,0,0,0,0,0,0]
        
        ### If hand detected

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=3, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=3, circle_radius=2),
                                        )
                    
                # Render left or right detection
                #if get_label(num, hand, results):
                #    text, coord = get_label(num, hand, results)
                #    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 1, cv2.LINE_AA)
            
            hand_status = True

            ### Measure Angle 

            # Draw angles to image from joint list
            #draw_finger_angles(frame, results, joint_list_0)
            #draw_finger_angles(frame, results, joint_list_1)
            #draw_finger_angles(frame, results, joint_list_2)
            #draw_finger_angles(frame, results, joint_list_3)
            #draw_finger_angles(frame, results, joint_list_4)

            angle_0 = get_finger_angles(results, joint_list_0)
            angle_1 = get_finger_angles(results, joint_list_1)
            angle_2 = get_finger_angles(results, joint_list_2)
            angle_3 = get_finger_angles(results, joint_list_3)
            angle_4 = get_finger_angles(results, joint_list_4)

            timer = round(time.time()-start_time,2)/2

            hand_angle = [  angle_0[0], angle_0[1], angle_0[2],
                            angle_1[0], angle_1[1], angle_1[2],
                            angle_2[0], angle_2[1], angle_2[2],   
                            angle_3[0], angle_3[1], angle_3[2],
                            angle_4[0], angle_4[1], angle_4[2] ]

            writer_ang.writerow([   timer, 
                                    angle_0[0], angle_0[1], angle_0[2],
                                    angle_1[0], angle_1[1], angle_1[2],
                                    angle_2[0], angle_2[1], angle_2[2],   
                                    angle_3[0], angle_3[1], angle_3[2],
                                    angle_4[0], angle_4[1], angle_4[2] ])


            ### Measure Distance
            
            # Create new variabel for wrist 
            wrist = np.array( [hand.landmark[9].x, hand.landmark[9].y] )

            timer = round(time.time()-start_time,2)/2
            writer_eye.writerow([timer, wrist[0], wrist[1], red_x/vwidth, red_y/vheight ])

            # Create new variabel for fingertip
            tip_0 = np.array([hand.landmark[4].x, hand.landmark[4].y] ) # , hand.landmark[4].z
            tip_1 = np.array([hand.landmark[8].x, hand.landmark[8].y] ) # , hand.landmark[8].z
            tip_2 = np.array([hand.landmark[12].x, hand.landmark[12].y] ) # , hand.landmark[12].z
            tip_3 = np.array([hand.landmark[16].x, hand.landmark[16].y] ) # , hand.landmark[16].z
            tip_4 = np.array([hand.landmark[20].x, hand.landmark[20].y] ) # , hand.landmark[20].z
            
            # Drawing circle in fingertip
            frame = cv2.circle(frame, ( int (hand.landmark[4].x * vwidth), 
                                        int (hand.landmark[4].y * vheight)), 
                                        radius=10, color=(0, 0, 255), thickness=-1)
            
            frame = cv2.circle(frame, ( int (hand.landmark[8].x * vwidth), 
                                        int (hand.landmark[8].y * vheight)), 
                                        radius=10, color=(0, 0, 255), thickness=-1)

            frame = cv2.circle(frame, ( int (hand.landmark[12].x * vwidth), 
                                        int (hand.landmark[12].y * vheight)), 
                                        radius=10, color=(0, 0, 255), thickness=-1)
            
            frame = cv2.circle(frame, ( int (hand.landmark[16].x * vwidth), 
                                        int (hand.landmark[16].y * vheight)), 
                                        radius=10, color=(0, 0, 255), thickness=-1)
            
            frame = cv2.circle(frame, ( int (hand.landmark[20].x * vwidth), 
                                        int (hand.landmark[20].y * vheight)), 
                                        radius=10, color=(0, 0, 255), thickness=-1)
        ### If hand NOT detected
        else:
            # Store CSV without hand data
            timer = round(time.time()-start_time,2)/2
            wrist = [0,0]       #wrist = [None,None]
            d_wrist_comp = 0    # new
            #speed_hand = 0
            writer_eye.writerow([timer, wrist[0], wrist[1], red_x/vwidth, red_y/vheight ])



        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Neo4j: ADD TIME & PLACE NODE
                
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        if toc_0 != current_time:   
            toc_0 = current_time  
        
            # Create TIME Interaction 
            time_now = ''
            time_minute = int(now.hour)*60 + int(now.minute)

            if (now.hour < 4):
                time_now = 'mid_night'
            elif (now.hour < 6):
                time_now = 'early_morning'
            elif (now.hour < 10):
                time_now = 'morning'
            elif (now.hour < 12):
                time_now = 'late_morning'
            elif (now.hour < 15 ):
                time_now = 'afternoon'
            elif (now.hour < 17 ):
                time_now = 'late_afternoon'
            elif (now.hour < 21 ):
                time_now = 'evening'
            elif (now.hour < 24 ):
                time_now = 'night'

            if time_now not in time_string:
                time_string.append(time_now)
                query_string = 'CREATE (' + str(time_now) + ':Time {name:"' + str(time_now) + '"})'
                conn.query(query_string, db='neo4j')

                query_string = 'MATCH (a:Time {name:"' + str(time_now) + '"})  OPTIONAL MATCH (b:Person {name:"person"}) CREATE (a)-[r:IN_TIME {'  + 'weight:0' + '}]->(b) return r'
                conn.query(query_string, db='neo4j')

                time_interact_string = []

            # Create PLACE Interaction 
            location_now = ''
            location_sensor = 5

            if (location_sensor == 1):
                location_now = 'entrance'
            elif (location_sensor == 2):
                location_now = 'kitchen'
            elif (location_sensor == 3):
                location_now = 'toilet'
            elif (location_sensor == 4):
                location_now = 'bathroom'
            elif (location_sensor == 5):
                location_now = 'living_room'
            elif (location_sensor == 6):
                location_now = 'bedroom'
            elif (location_sensor == 7):
                location_now = 'outside'
            else:
                location_now = 'unknown'

            if location_now not in location_string:
                location_string.append(location_now)
                query_string = 'CREATE (' + str(location_now) + ':Location {name:"' + str(location_now) + '"})'
                conn.query(query_string, db='neo4j')

                query_string = 'MATCH (a:Location {name:"' + str(location_now) + '"})  OPTIONAL MATCH (b:Person {name:"person"}) CREATE (a)-[r:IN_LOCATION {'  + 'weight:0' + '}]->(b) return r'
                conn.query(query_string, db='neo4j')



        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Objects Detection

        # Get Results
        results_obj = model(frame)
        df_tracked_objects = results_obj.pandas().xyxy[0]
        list_tracked_objects = df_tracked_objects.values.tolist()

        if len(list_tracked_objects) > 0:
            
            # List declaration
            object_detected = []
            d_attention_list = []
            d_diagonal_list = []
            
            name_object_list = []
            objects_inter_all = []

            for x1, y1, x2, y2, conf_pred, cls_id, cls in list_tracked_objects:

                if cls == "cup": # or cls == "spoon" or cls == "fork" or cls == "knife" or cls == "laptop" or cls == "mouse":
                    # or cls == "person":

                    center_x = int ((x1+x2)/2)
                    center_y = int ((y1+y2)/2)

                    # Draw objects features
                    cv2.circle(frame, (center_x, center_y), radius=10, color=(0, 0, 255), thickness=-1)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(frame, cls , (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Draw line objects to attention
                    cv2.line(frame, (center_x, center_y), (red_x, red_y), color=(0, 0, 255), thickness=3)

                    name_object_list.append(cls)

                    #if o_move == 0:
                    #    initial_x = center_x
                    #    initial_y = center_y
                    #    o_move = 1
                    #else:
                    #    o_move = int (sqrt(pow(center_x-initial_x,2) + pow(center_y-initial_y,2)))
                    #    initial_x = center_x
                    #    initial_y = center_y

                    #a1 = int (x1 - center_x)
                    #b1 = int (y1 - center_y)
                    #a2 = int (x1+box_w - center_x)
                    #b2 = int (y1+box_h - center_y)

                    o_length = int(x2) - int(x1)
                    o_width = int(y2) - int(y1)

                    o_diagonal = int(sqrt(pow(o_length,2) + pow(o_width,2)))
                    d_diagonal_list.append(o_diagonal)


                    # Detect Hand and Object -> measure speed

                    if hand_status:
                        
                        cv2.line(frame, (center_x, center_y), (int(wrist[0]*vwidth), int(wrist[1]*vheight)), color=(255, 0, 0), thickness=3)
                        
                        d_wrist = int (sqrt(pow(center_x-int(wrist[0]*vwidth),2) + pow(center_y-int(wrist[1]*vheight),2)))

                        if d_wrist_comp == 0:
                            d_wrist_comp = d_wrist
                            d_attention_comp = d_attention
                            
                            timer = round(time.time()-start_time,2)/2
                            d_timer_comp = timer
                            #print("d_timer_comp: " + str(d_timer_comp))
                        
                        else:
                            #>>>>>>>>>>>>>>>>>>>> trial
                            timer = round(time.time()-start_time,2)/2

                            if (d_timer_comp - timer) != 0:
                                speed_hand = round(abs(d_wrist_comp - d_wrist) / abs(d_timer_comp - timer))
                                speed_eye = round(abs(d_attention_comp - d_attention) / abs(d_timer_comp - timer))

                                #print("Selisih timer:" + str(d_timer_comp - timer))
                                #print(speed_hand, speed_eye)
                                d_wrist_comp = d_wrist
                                d_timer_comp = timer
                                d_attention_comp = d_attention

                    else:
                        d_wrist_comp = 0
                        #speed_hand = 0
                        #speed_eye = 0
                    
                    # angle_wrist = math.atan( (int(wrist[1]*vheight) - y_center) / (int(wrist[0]*vwidth) - x_center) )
                    # angle_finger = math.atan( (int(wrist[1]*vheight) - y_center) / (int(wrist[0]*vwidth) - x_center) )

                    # Count 4 fingetip distance to fingertip tumb
                    d_attention = int (sqrt(pow(center_x-red_x,2) + pow(center_y-red_y,2)))
                    d_attention_list.append(d_attention)

                    objects_inter_all.append(cls)
                    objects_inter_all = list(set(objects_inter_all))

                    objects_name_all.append(cls)
                    objects_name_all = list(set(objects_name_all))

                    writer_speed.writerow([ timer, d_attention, d_wrist, speed_eye, speed_hand])
                
                #Hand and Object is NOT Detected
                elif results.multi_hand_landmarks is None :
                    
                    pass
                    #d_wrist = 0
                    #speed_hand = 0
        


            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Neo4j: CREATE GRAPH
                
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            if toc_1 != current_time:   
                toc_1 = current_time   
                
                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Neo4j: CREATE OBJECT NODE

                for i in range(len(objects_name_all)):
                    if str(objects_name_all[i]) not in objects_name_saved:
                        objects_name_saved.append(str(objects_name_all[i]))
                        query_string = 'CREATE (' + str(objects_name_all[i]) + ':Object {name:"' + str(objects_name_all[i]) + '", weight:0})'
                        conn.query(query_string, db='neo4j')
                    else:
                        pass

                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Neo4j: CREATE OBJECT TO OBJECT EDGE  
                
                if (len(objects_inter_all) > 1):
                    relation = list(itertools.combinations(objects_inter_all, 2))

                    #print(relation)

                    for i in range(len(relation)):
                        if str(sorted(relation[i])) not in objects_inter_string:
                            objects_inter_string.append(str(sorted(relation[i])))
                            query_string = 'MATCH (a:Object {name:"' + str(relation[i][0]) + '"}) OPTIONAL MATCH (b:Object {name:"' + str(relation[i][1]) + '"}) CREATE (a)-[r:NEAR {'  + 'weight:0' + '}]->(b) return r'
                            conn.query(query_string, db='neo4j')
                        else:
                            query_string = 'MATCH (a:Object {name:"' + str(relation[i][0]) + '"})-[r]->(b:Object {name:"' + str(relation[i][1]) + '"}) SET r.weight = r.weight + 1' 
                            conn.query(query_string, db='neo4j')

                    #print(objects_inter_all)
        else:
            d_wrist = 0
            speed_hand = 0

        cv2.circle(frame, (red_x, red_y), radius=10, color=(0, 0, 255), thickness=-1)




        # Show Object Focused
        object_focus = min(d_attention_list, default="EMPTY")
            
        if object_focus != "EMPTY":
            object_focus_id = d_attention_list.index(object_focus)

            if (d_attention_list[ object_focus_id ] < (d_diagonal_list[ object_focus_id ]/2) ) :
                str_object_focus = str(name_object_list[ object_focus_id ])
                cv2.putText(frame, str_object_focus + " (with attention)", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)

                object_temp = str_object_focus

                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Neo4j: ADD OBJECT WEIGHT

                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
            
                if toc_2 != current_time:    
                    print(current_time + " : Attention to " + str_object_focus)
                    toc_2 = current_time
                    
                    query_string = 'MATCH (a:Object {name:"' + str_object_focus + '"}) SET a.weight = a.weight + 1' 
                    conn.query(query_string, db='neo4j')

            else:
                cv2.putText(frame, object_temp, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)
        else:
            cv2.putText(frame, object_temp, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)



        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Hand Action Detection 

        if hand_status == True: # object_status == True 
            
            # Interaction focus on attention
            focus_x = center_x #red_x
            focus_y = center_y #red_y

            # Draw 5 fingetip to attention                        
            #cv2.line(frame, (focus_x, focus_y), (int(tip_0[0]*vwidth), int(tip_0[1]*vheight)), color=(0, 255, 0), thickness=2)
            #cv2.line(frame, (focus_x, focus_y), (int(tip_1[0]*vwidth), int(tip_1[1]*vheight)), color=(0, 200, 0), thickness=2)
            #cv2.line(frame, (focus_x, focus_y), (int(tip_2[0]*vwidth), int(tip_2[1]*vheight)), color=(0, 150, 0), thickness=2)
            #cv2.line(frame, (focus_x, focus_y), (int(tip_3[0]*vwidth), int(tip_3[1]*vheight)), color=(0, 100, 0), thickness=2)
            #cv2.line(frame, (focus_x, focus_y), (int(tip_4[0]*vwidth), int(tip_4[1]*vheight)), color=(0, 50, 0), thickness=2)

            # Draw 4 fingetip to tumb-tip    
            #cv2.line(frame, (int(tip_0[0]*vwidth), int(tip_0[1]*vheight)) , (int(tip_1[0]*vwidth), int(tip_1[1]*vheight)), color=(0, 200, 0), thickness=2)
            #cv2.line(frame, (int(tip_0[0]*vwidth), int(tip_0[1]*vheight)) , (int(tip_2[0]*vwidth), int(tip_2[1]*vheight)), color=(0, 150, 0), thickness=2)
            #cv2.line(frame, (int(tip_0[0]*vwidth), int(tip_0[1]*vheight)) , (int(tip_3[0]*vwidth), int(tip_3[1]*vheight)), color=(0, 100, 0), thickness=2)
            #cv2.line(frame, (int(tip_0[0]*vwidth), int(tip_0[1]*vheight)) , (int(tip_4[0]*vwidth), int(tip_4[1]*vheight)), color=(0, 50, 0), thickness=2)

            # Count fingetip distance to attention
            tip_0[0] = int((tip_0[0] * vwidth) - focus_x)
            tip_0[1] = int((tip_0[1] * vheight) - focus_y)
            tip_1[0] = int((tip_1[0] * vwidth) - focus_x)
            tip_1[1] = int((tip_1[1] * vheight) - focus_y)
            tip_2[0] = int((tip_2[0] * vwidth) - focus_x)
            tip_2[1] = int((tip_2[1] * vheight) - focus_y)
            tip_3[0] = int((tip_3[0] * vwidth) - focus_x)
            tip_3[1] = int((tip_3[1] * vheight) - focus_y)
            tip_4[0] = int((tip_4[0] * vwidth) - focus_x)
            tip_4[1] = int((tip_4[1] * vheight) - focus_y)

            d_tip_0 = int (sqrt(pow(tip_0[0],2) + pow(tip_0[1],2)))
            d_tip_1 = int (sqrt(pow(tip_1[0],2) + pow(tip_1[1],2)))
            d_tip_2 = int (sqrt(pow(tip_2[0],2) + pow(tip_2[1],2)))
            d_tip_3 = int (sqrt(pow(tip_3[0],2) + pow(tip_3[1],2)))
            d_tip_4 = int (sqrt(pow(tip_4[0],2) + pow(tip_4[1],2)))

            # Count 4 fingetip distance ke fingertip tumb
            d_pinch_1 = int (sqrt(pow(tip_0[0]-tip_1[0],2) + pow(tip_0[1]-tip_1[1],2)))
            d_pinch_2 = int (sqrt(pow(tip_0[0]-tip_2[0],2) + pow(tip_0[1]-tip_2[1],2)))
            d_pinch_3 = int (sqrt(pow(tip_0[0]-tip_3[0],2) + pow(tip_0[1]-tip_3[1],2)))
            d_pinch_4 = int (sqrt(pow(tip_0[0]-tip_4[0],2) + pow(tip_0[1]-tip_4[1],2)))

            hand_status = False

            timer = round(time.time()-start_time,2)/2
            hand_position = [d_tip_0, d_tip_1, d_tip_2, d_tip_3, d_tip_4, d_pinch_1, d_pinch_2, d_pinch_3, d_pinch_4]

            row_id = str(group_id) + "_" + str(measurement_id)

            writer_pos.writerow([
                                timer, row_id, series_id, measurement_id, group_id, surface, 
                                #o_move, o_length, o_width, 
                                d_tip_0, d_tip_1, d_tip_2, d_tip_3, d_tip_4, 
                                d_pinch_1, d_pinch_2, d_pinch_3, d_pinch_4 ])
            
            ### PREDICT ACTION

            # Feature Scaling
            coll_hand.append(hand_position)

            if len(coll_hand) == sequence_data:
                x_data = np.array(list(coll_hand))
                x_train = sc_input.transform(x_data)
                x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
                x_train = x_train[None, :, :]
                           
                outputs = model_nn(x_train)
                confidence, predicted = torch.max(outputs.data, 1)
                
                str_conf = str( format(confidence.item()*10,".2f") )

                if predicted.item() == 0:
                    cv2.putText(frame, "grasp " + str_conf + "%", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)
                elif predicted.item() == 1:
                    cv2.putText(frame, "reach " + str_conf + "%", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)
                elif predicted.item() == 2:
                    cv2.putText(frame, "release " + str_conf + "%", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)
                elif predicted.item() == 3:
                    cv2.putText(frame, "wonder " + str_conf + "%", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)
       

                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Neo4j: ADD PERSON TO OBJECT WEIGHT
                 
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")

                if toc_3 != current_time:    
                    print(current_time + " : Hand interract with " + str_object_focus)
                    toc_3 = current_time

                    str_object_grasp = str_object_focus

                    if str_object_grasp not in objects_grasp_all:

                        query_string = 'MATCH (a:Person {name:"person"}) OPTIONAL MATCH (b:Object {name:"' + str(str_object_grasp) + '"}) CREATE (a)-[r:INTERACT {'  + 'weight:0' + '}]->(b) return r'
                        conn.query(query_string, db='neo4j')

                        objects_grasp_all.append(str_object_grasp)
                        objects_grasp_all = list(set(objects_grasp_all))

                    else:
                        query_string = 'MATCH (a:Person {name:"person"})-[r:INTERACT]->(b:Object {name:"' + str(str_object_grasp) + '"}) SET r.weight = r.weight + 1' 
                        conn.query(query_string, db='neo4j')


        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Neo4j: ADD TIME NODE to PERSON AND OBJECT
                    
        else:
            #No Hand Detected
            cv2.putText(frame, "no interaction with", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)
            
            #speed_eye = 0
            speed_hand = 0
            #hand_position = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            #coll_hand.append(hand_position)

            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Save data every 1 second to Neo4j
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            if toc_4 != current_time:    
                print(current_time + " : No attention to any object")
                toc_4 = current_time
            
            #str_object_focus = "unknown object"
             

        ############################################################# Save to CSV and Send Data via UDP

        timer = round(time.time()-start_time,2)/2

        writer_speed.writerow([ timer, d_attention, d_wrist, speed_eye, speed_hand])
        
        std_eye = d_attention/vwidth 
        std_hand = d_wrist/vwidth 
           
        # Send Data via UDP
        #package = str(timer) + "/" + str(std_red_x) + "/" + str(speed_eye/30000) + "/" + str(std_wrist_x) + "/" + str(speed_hand/3000)
        package = str(timer) + "/" + str(std_eye) + "/" + str(std_hand) + "/" + str(speed_eye/15000) + "/" + str(speed_hand/1500)
        
        MESSAGE = package.encode()
        #MESSAGE = str(red_x/vwidth).encode()
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))

        
        
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Show the video/camera streaming 

        cv2.imshow('Stream', frame)
        #cv2.imshow('Stream', np.squeeze(results_obj.render()))

        #if cv2.waitKey(10) & 0xFF == ord('q'):
        #    break

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

print("-------------------------------------------------- END")

totaltime = time.time()-starttime
#>>>>>>>>>>>> bug
print(frames, "frames", frames/totaltime, "frame/second")

cap.release()
cv2.destroyAllWindows()
