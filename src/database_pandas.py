''' This file will contain code to store inferred faces in a pandas dataframe

    The dataframe will have the following columns:
    1. name/id
    2. Time
    3. Date
    4. Location
    5. Face Distance - the distance between the face and the known face (which can be considered
    as inverse of confidence)

    The maximum number of rows in the dataframe will be equal to 1 million

    The dataframe will be stored in a csv file

Author: Anubhav
Date: 24/02/2023
'''

import time
import pandas as pd
from parameters import REPORT_PATH

df_inferred_faces = pd.DataFrame(columns=['name/id','time','date','location','face_distance'])


def store_inferred_face_in_dataframe(name_of_person, face_distance):
    '''
        This function will store the name of the person and the face distance in the dataframe

        Arguments:
            name_of_person {string} -- name of the person
            face_distance {float} -- face distance
        
        Returns:
            None
    '''

    #current time in HH:MM:SS format
    current_time = time.strftime("%H:%M:%S")
    #current date in DD/MM/YYYY format
    current_date = time.strftime("%d/%m/%Y")
    #current location
    current_location = 'GRIL Lab'

    # We want to store the name of the person only if it is not already present in the dataframe
    if name_of_person not in df_inferred_faces['name/id'].values:        
        #store the name of the person in the dataframe
        df_inferred_faces.loc[len(df_inferred_faces)] = [name_of_person, current_time, current_date, current_location, face_distance]

    #if the number of rows in the dataframe is greater than 1 million, then save the dataframe in a csv file
    #if len(df_inferred_faces) > 1000000:
    #    df_inferred_faces.to_csv('inferred_faces.csv')
    #    df_inferred_faces = pd.DataFrame(columns=['name/id','time','date','location','face_distance'])


def store_dataframe_in_csv():
    '''
        This function will store the dataframe in a csv file

        Arguments:
            None
        
        Returns:
            True if the dataframe is stored in the csv file, False otherwise
    '''
    global df_inferred_faces

    try:
        df_inferred_faces.to_csv(REPORT_PATH, encoding='utf-8')
        print('Dataframe stored in csv file')
        return True
    except Exception as e:
        print(f'Exception occured while storing dataframe in csv file: {e}')
        return False
    finally:
        #reset the dataframe
        #df_inferred_faces = pd.DataFrame(columns=['name/id','time','date','location','face_distance'])
        pass