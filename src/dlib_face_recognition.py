# -*- coding: utf-8 -*-
"""
@Original Author: abhilash
@Modified by: Anubhav Patrick
"""

#importing the required libraries
import pickle
import cv2
import face_recognition
import numpy as np

from database_pandas import store_inferred_face_in_dataframe
from parameters import NUMBER_OF_TIMES_TO_UPSAMPLE, DLIB_FACE_ENCODING_PATH

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(DLIB_FACE_ENCODING_PATH,"rb").read())

#save the encodings and the corresponding labels in seperate arrays in the same order
known_face_encodings = data["encodings"]
known_face_names = data["names"]

#initialize the array variable to hold all face locations, encodings and names 
all_face_locations = []
all_face_encodings = []
all_face_names = []
all_processed_frames = []


def single_frame_face_recognition(frame, frame_downsample, number_of_times_to_upsample, model, face_matching_tolerance):
    '''Single frame face recognition function
    
    Arguments:
        frame {numpy array} -- frame to be processed
        frame_downsample {bool} -- whether to downsample the frame or not
        number_of_times_to_upsample
        model -- face detection model
        face_matching_tolerance -- tolerance for face matching
    
    Returns:
        a processed frame
    '''

    if frame_downsample:
        #resize the current frame to 1/4 size to proces faster
        #current_frame_small = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        
        #resize the frame to 1024x576 to display the video if frame is too big
        # frame.shape[1] is width and frame.shape[0] is height
        if frame.shape[1] > 1024 or frame.shape[0] > 576:
            current_frame_small = cv2.resize(frame,(1024,576))
        else:
            current_frame_small = frame
    else:
        #consider the frame as it is
        current_frame_small = frame

    #detect all faces in the image
    #arguments are image,no_of_times_to_upsample, model
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample,model)
        
    #detect face encodings for all the faces detected
    all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)

    #looping through the face locations and the face embeddings
    for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
        #splitting the tuple to get the four position values of current face
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        
        if frame_downsample:
            #change the position maginitude to fit the actual size video frame
            '''top_pos = top_pos*4
            right_pos = right_pos*4
            bottom_pos = bottom_pos*4
            left_pos = left_pos*4'''
            pass
        
        #find all the matches and get the list of matches
        all_matches = face_recognition.face_distance(known_face_encodings, current_face_encoding)
        # Find the best match (smallest distance to a known face)
        best_match_index = np.argmin(all_matches)
        # If the best match is within tolerance, use the name of the known face
        if all_matches[best_match_index] <= face_matching_tolerance:
            name_of_person = known_face_names[best_match_index]
            #save the name of the person in the dataframe
            store_inferred_face_in_dataframe(name_of_person, all_matches[best_match_index])
        else:
            name_of_person = 'Unknown face'

        # For known face use green color and for unknown face use red color
        if name_of_person == 'Unknown face':
            color = (0,0,255) #Red
        else:
            color = (0,255,0) #Green
        
        #draw rectangle around the face    
        cv2.rectangle(current_frame_small,(left_pos,top_pos),(right_pos,bottom_pos),color,5)
        
        #display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame_small, name_of_person, (left_pos,bottom_pos), font, 1, (255,255,255),2)
    
    yield current_frame_small



def multi_frame_face_recognition(frames_buffer, frame_downsample=True, number_of_times_to_upsample=1, model='hog', face_matching_tolerance=0.45):
    '''
        Multi frame face recognition function

        Arguments:
            frames_buffer {list} -- list of frames to be processed
            frame_downsample {bool} -- whether to downsample the frame or not
            number_of_times_to_upsample
            model {string} -- model to be used for face detection
            face_matching_tolerance {float} -- tolerance for face matching

        Yields:
            processed frames
    '''
    #pop first frame from frames_buffer to get the first frame
    while True:
        if len(frames_buffer) > 0:
            _ = frames_buffer.pop(0)
            break

    # Continue looping until there are no more frames to process.
    while True:
        
        #if is_stream:
          # check if there are frames in the buffer

        if len(frames_buffer) > 0:
          #pop first frame from frames_buffer 
          img0 = frames_buffer.pop(0)
          if img0 is None:
            continue
          ret = True #we have successfully read one frame from stream
          if len(frames_buffer) >= 10:
            frames_buffer.clear() #clear the buffer if it has more than 10 frames to avoid memory overflow
        else:
          # buffer is empty, nothing to do
          continue

        #if we are able to read a frame then process it
        if ret:
            #yield the processed frame to the main thread
            yield from single_frame_face_recognition(frame= img0, 
                                                     frame_downsample=frame_downsample, 
                                                     number_of_times_to_upsample=NUMBER_OF_TIMES_TO_UPSAMPLE, 
                                                     model=model, 
                                                     face_matching_tolerance=face_matching_tolerance)



