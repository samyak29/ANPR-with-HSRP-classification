# test.py

import numpy as np
import os
import tensorflow as tf
import cv2
import OCR

from utils import label_map_util
from utils import visualization_utils as vis_util
from distutils.version import StrictVersion

# module level variables ##############################################################################################
TEST_IMAGE_DIR = os.getcwd() + "/3" + "/test_images"
FROZEN_INFERENCE_GRAPH_LOC = os.getcwd() + "/3" + "/inference_graph/frozen_inference_graph.pb"
LABELS_LOC = os.getcwd() + "/3" + "/" + "label_map.pbtxt"
NUM_CLASSES = 1
RETRAINED_LABELS_TXT_FILE_LOC = os.getcwd() + "/2" + "/" + "retrained_labels.txt"
RETRAINED_GRAPH_PB_FILE_LOC = os.getcwd() + "/2" + "/" + "retrained_graph.pb"
results_dir = os.getcwd() + "/results/"
SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)
#######################################################################################################################
def main():
    print("starting program . . .")

    if not checkIfNecessaryPathsAndFilesExist():
        return
    # end if
    classifications = []
    # for each line in the label file . . .
    for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
        # remove the carriage return
        classification = currentLine.rstrip()
        # and append to the list
        classifications.append(classification)
    # this next comment line is necessary to avoid a false PyCharm warning
    # noinspection PyUnresolvedReferences
    if StrictVersion(tf.__version__) < StrictVersion('1.5.0'):
        raise ImportError('Please upgrade your tensorflow installation to v1.5.* or later!')
    # end if

    # load a (frozen) TensorFlow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        with tf.gfile.GFile(FROZEN_INFERENCE_GRAPH_LOC, 'rb') as fid:
            od_graph_def = tf.GraphDef()
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        # end with
    # end with
    classification_graph = tf.Graph()
    with classification_graph.as_default():
        with tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
            # instantiate a GraphDef object
            graphDef = tf.GraphDef()
            # read in retrained graph into the GraphDef object
            graphDef.ParseFromString(retrainedGraphFile.read())
            # import the graph into the current default Graph, note that we don't need to be concerned with the return value
            tf.import_graph_def(graphDef, name='')
    # end with
    # Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(LABELS_LOC)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)



    with tf.Session(graph=detection_graph) as sessd:
        with tf.Session(graph=classification_graph) as sessc:
            for imageFileName in os.listdir(TEST_IMAGE_DIR):
                if not imageFileName.endswith(".jpg"):
                    continue
                image_path = TEST_IMAGE_DIR + "/" + imageFileName
                print(image_path)
                image_np = cv2.imread(image_path)

                if image_np is None:
                    print("error reading file " + image_path)
                    continue
                # end if

                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Expanded dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sessd.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                img_height, img_width, img_channel = image_np.shape
                THRESHOLD = 0.4
                #print(scores[0])
                i = 0
                while scores[0][i] >= THRESHOLD and i <= 99:
                    ymin, xmin, ymax, xmax = boxes[0][i]
                    x_up = int(xmin * img_width)
                    y_up = int(ymin * img_height)
                    x_down = int(xmax * img_width)
                    y_down = int(ymax * img_height)
                    cropped_image = image_np[y_up:y_down, x_up:x_down]
                    detected_ocr = OCR.ocr(cropped_image)
                    # get the final tensor from the graph
                    finalTensor = sessc.graph.get_tensor_by_name('final_result:0')
                    # convert the OpenCV image (numpy array) to a TensorFlow image
                    tfImage = np.array(cropped_image)[:, :, 0:3]
                    # run the network to get the predictions
                    predictions = sessc.run(finalTensor, {'DecodeJpeg:0': tfImage})
                    # sort predictions from most confidence to least confidence
                    sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

                    prediction = sortedPredictions[0]
                    strClassification = classifications[prediction]
                    if strClassification.endswith("s"):
                        strClassification = strClassification[:-1]
                    confidence = predictions[0][prediction]
                    scoreAsAPercent = confidence * 100.0
                    label = strClassification + ", " + "{0:.2f}".format(scoreAsAPercent)
                    cv2.putText(image_np, label,(int(x_up),int(y_down)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1)
                    #cv2.putText(image_np, detected_ocr, (int((x_down+x_up)/2), int(y_up), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 0, 0),1)
                    i = i+1
                    #cropped_image_path = os.getcwd() + "/detected_number_plates/" + imageFileName[:-4] + str(i) + ".jpg"
                    #cv2.imwrite(cropped_image_path, cropped_image)

                vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=3)

                # cv2.imshow("image_np", image_np)
                # while cv2.waitKey() != 32:
                #     pass
                cv2.imwrite(results_dir + imageFileName , image_np)


#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(TEST_IMAGE_DIR):
        print('ERROR: TEST_IMAGE_DIR "' + TEST_IMAGE_DIR + '" does not seem to exist')
        return False
    # end if

    # ToDo: check here that the test image directory contains at least one image

    if not os.path.exists(FROZEN_INFERENCE_GRAPH_LOC):
        print('ERROR: FROZEN_INFERENCE_GRAPH_LOC "' + FROZEN_INFERENCE_GRAPH_LOC + '" does not seem to exist')
        print('was the inference graph exported successfully?')
        return False
    # end if

    if not os.path.exists(LABELS_LOC):
        print('ERROR: the label map file "' + LABELS_LOC + '" does not seem to exist')
        return False
    # end if
    if not os.path.exists(RETRAINED_LABELS_TXT_FILE_LOC):
        print('ERROR: RETRAINED_LABELS_TXT_FILE_LOC "' + RETRAINED_LABELS_TXT_FILE_LOC + '" does not seem to exist')
        return False
        # end if

    if not os.path.exists(RETRAINED_GRAPH_PB_FILE_LOC):
        print('ERROR: RETRAINED_GRAPH_PB_FILE_LOC "' + RETRAINED_GRAPH_PB_FILE_LOC + '" does not seem to exist')
        return False
    # end if

    return True
if __name__ == "__main__":
    main()
