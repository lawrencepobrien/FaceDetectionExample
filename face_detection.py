from mtcnn.mtcnn import MTCNN

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from numpy import asarray

import urllib.request

def store_image(url, local_file_name):
  with urllib.request.urlopen(url) as resource:
    with open(local_file_name, 'wb') as f:
      f.write(resource.read())

size = (224, 224)
face_detector = MTCNN()

store_image('https://ichef.bbci.co.uk/news/320/cpsprodpb/5944/production/_107725822_55fd57ad-c509-4335-a7d2-bcc86e32be72.jpg',
            'iacocca_1.jpg')
store_image('https://www.gannett-cdn.com/presto/2019/07/03/PDTN/205798e7-9555-4245-99e1-fd300c50ce85-AP_080910055617.jpg?width=540&height=&fit=bounds&auto=webp',
            'iacocca_2.jpg')

def highlight_faces(image_path, faces):
    ax = plt.gca()
    
    for face in faces:
        x, y, width, height = face['box']
        rect = Rectangle((x, y), width, height, fill = False, color = 'red')
        ax.add_patch(rect)
    
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.show()
    
def extract_faces(image_path):
    image = plt.imread(image_path)
    faces = face_detector.detect_faces(image)
    
    face_pics = []
    
    for face in faces:
        x1, y1, width, height = face['box']
        x2 = x1 + width
        y2 = y1 + height
        
        bounding_rect = image[y1:y2, x1:x2]
        
        face_pic = Image.fromarray(bounding_rect).resize(size)
        face_pics.append(asarray(face_pic))
        
    return face_pics
        
extracted_face = extract_faces('iacocca_1.jpg')
faces = face_detector.detect_faces(plt.imread('iacocca_1.jpg'))
rect_imgs = highlight_faces('iacocca_1.jpg', faces)

plt.imshow(extracted_face[0])
plt.show()