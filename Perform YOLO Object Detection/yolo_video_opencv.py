# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True,
	help='path to input video')
ap.add_argument('-o', '--output', required=True,
	help='path to output video')
ap.add_argument('-y', '--yolo', required=True,
	help='base path to YOLO directory')
ap.add_argument('-c', '--confidence', type=float, default=0.5,
	help='minimum probability to filter weak detections')
ap.add_argument('-t', '--threshold', type=float, default=0.3,
	help='threshold when applying non-maxima suppression')
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labels_path = os.path.sep.join([args['yolo'], 'coco.names'])
labels = open(labels_path).read().strip().split('\n')

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# derive the paths to the YOLO weights and model configuration
weights_path = os.path.sep.join([args['yolo'], 'yolov3.weights'])
config_path = os.path.sep.join([args['yolo'], 'yolov3.cfg'])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print('Loading YOLO from disk...')
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args['input'])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	number_of_frames = int(vs.get(prop))
	print('{} total frames in video'.format(number_of_frames))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print('Could not determine # of frames in video')
	print('No approx. completion time can be provided')
	number_of_frames = -1

scale_factor = 1 / 255.0
new_size = (416, 416)

while True:
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, scale_factor, new_size, swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(layer_names)
	end = time.time()

	boxes, confidences, class_ids = [], [], []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > args['confidence']:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype('int')
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				class_ids.append(class_id)

	# apply non-maxima suppression to suppress weak, overlapping bounding boxes
	nmsboxes = cv2.dnn.NMSBoxes(boxes, confidences, args['confidence'], args['threshold'])

	if len(nmsboxes) > 0:
		for i in nmsboxes.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in colors[class_ids[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = '{}: {:.4f}'.format(labels[class_ids[i]], confidences[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		writer = cv2.VideoWriter(args['output'], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if number_of_frames > 0:
			elapse = (end - start)
			print('Single frame took {:.2f} seconds'.format(elapse))
			print('Estimated total time to finish: {:.2f} seconds'.format(
				elapse * number_of_frames))

	writer.write(frame)

print('Cleaning up...')
writer.release()
vs.release()