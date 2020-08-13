import cv2
import numpy as np
import time
import argparse
import os

CONFIDENCE_THRESHOLD = .5
IOU_THRESHOLD = .5

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
	help='Path to input image')
ap.add_argument('-y', '--yolo', required=True,
	help='Base path to YOLO directory')
ap.add_argument('-c', '--confidence', type=float, default=CONFIDENCE_THRESHOLD,
	help='Minimum probability to filter weak detections')
ap.add_argument('-t', '--threshold', type=float, default=IOU_THRESHOLD,
	help='Threshold when applying non-maxima suppression')
args = vars(ap.parse_args())

labels_path = os.path.sep.join([args['yolo'], 'coco.names'])
config_path = os.path.sep.join([args['yolo'], 'yolov3.cfg']) # the neural network configuration
weights_path = os.path.sep.join([args['yolo'], 'yolov3.weights']) # the YOLO net weights file

labels = open(labels_path).read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# load the YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

image_name = args['image']
image = cv2.imread(image_name)
filename, ext = os.path.basename(image_name).split('.')

# normalize, scale, and reshape the image
(h, w) = image.shape[:2]
scale_factor = 1/255.0
new_size = (416, 416)
blob = cv2.dnn.blobFromImage(image, scale_factor, new_size, swapRB=True, crop=False) # 4D blob 

# print('image.shape:', image.shape)
# print('blob.shape:', blob.shape)

net.setInput(blob) # set the blob as the input of the network
# get all the layer names
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# feed forward and get the network output
# measure how much it took in seconds
start = time.perf_counter()
layer_outputs = net.forward(layer_names)
time_took = time.perf_counter() - start
print(f'Time took: {time_took:.2f}s')

# iterate over the neural network outputs 
# and discard any object that has the confidence 
# less than CONFIDENCE
boxes, confidences, class_ids = [], [], []
for output in layer_outputs:
    for detection in output:
        # extract the class id (label) and confidence (as a probability) of
        # the current object detection
        scores = detection[5:]
        class_id = np.argmax(scores) 
        confidence = scores[class_id]
        # discard weak predictions
        if confidence > args['confidence']:
            # scale the bounding box coordinates back relative to the
            # size of the image
            bounding_box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = bounding_box.astype('int') # return output of YOLO
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            # update
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# perform the non maximum suppression given the scores defined before
nmsboxes = cv2.dnn.NMSBoxes(boxes, confidences, args['confidence'], args['threshold'])

# draw a bounding box rectangle and label on the image 
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = .5
thickness = 2
if len(nmsboxes) > 0:
    for i in nmsboxes.flatten():
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
        text = f'{labels[class_ids[i]]}: {confidences[i]:.4f}'
        # calculate text width & height to draw the transparent boxes as background of the text
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
        text_offset_x = x
        text_offset_y = y - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y + text_height))
        cv2.rectangle(image, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
        overlay = image.copy()
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0) # add opacity (transparency to the box)
        cv2.putText(image, text, (text_offset_x, text_offset_y), font,
            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

cv2.imshow(filename, image)
cv2.waitKey(0)
cv2.imwrite('output/{}_YOLOv3.{}'.format(filename, ext), image)