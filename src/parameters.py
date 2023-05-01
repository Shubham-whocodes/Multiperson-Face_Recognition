'''
This module contain Parameters used by the app 
'''

#The default sender of emails
EMAIL_SENDER = 'cai20002@glbitm.ac.in'

#Path where dataset is stored
DATASET_PATH = '../dataset/train/pics_dlib_2/'

# The path where dlib face encodings are stored
DLIB_FACE_ENCODING_PATH = '../dataset/dlib_face_encoding.pkl'

# The path where face recognition report will be stored
REPORT_PATH = './static/reports/inferred_faces.csv'

# Files path where various supplementary files are stored
FILES_PATH = './static/files/'

# The path where the video will be stored
VIDEO_PATH = './static/files/vid1.mp4'

# The path where the video will be uploaded
VIDEO_UPLOAD_PATH = './static/video/vid2.mp4'

# face matching tolerance (distance -> less the distance, more the similarity)
FACE_MATCHING_TOLERANCE = 0.4

#face recognition model
FACE_RECOGNITION_MODEL = 'hog' #hog -> for CPU or cnn -> for GPU

# Number of times to upsample the image looking for faces
NUMBER_OF_TIMES_TO_UPSAMPLE = 1 # for realtime keep it to 1
