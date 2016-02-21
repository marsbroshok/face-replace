import skimage
import skimage.transform
import dlib
import cv2
import numpy as np

# Dlib face detector
detector = dlib.get_frontal_face_detector()

# Dlib landmarks predictor
predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

# OpenCV HAAR cascade
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')


def draw_str(dst, (x, y), s):
    """
    Copy of helper function from OpenCV common.py to draw a piece of text on the top of the image

    :param dst: image where text should be added
    :param (x, y): tuple of coordinates to place text to
    :param s: string of text
    :return: image with text
    """
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)


def _merge_images(img_top, img_bottom, mask=0):
    """
    Function to combine two images with mask by replacing all pixels of img_bottom which
    equals to mask by pixels from img_top.

    :param img_top: greyscale image which will replace masked pixels
    :param img_bottom: greyscale image which pixels will be replace
    :param mask: pixel value to be used as mask (int)
    :return: combined greyscale image
    """
    img_top = skimage.img_as_ubyte(img_top)
    img_bottom = skimage.img_as_ubyte(img_bottom)
    merge_layer = img_top == mask
    img_top[merge_layer] = img_bottom[merge_layer]
    return img_top


def _shape_to_array(shape):
    """
    Function to convert dlib shape object to array

    :param shape:
    :return:
    """
    return np.array([[p.x, p.y] for p in shape.parts()], dtype=float)


def _detect_face_dlib(image):
    """
    Function to detect faces in the input image with dlib

    :param image: grayscale image with face(s)
    :return: dlib regtangles object with detected face regions
    """
    return detector(image, 1)


def _detect_face_opencv(image, cascade):
    """
    Function to detect faces in the input image with OpenCV

    :param image: grayscale image with face(s)
    :param cascade: OpenCV CascadeClassifier object
    :return: array of detected face regions
    """
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return sorted(rects, key=lambda rect: rect[2] - rect[0], reverse=True)


def _array_to_dlib_rectangles(rects_array):
    """
    Function to convert array of rectangles (in format [[x1,y1,w1,h1],[x2,y2,w2,h2],...]) to dlib regtangles objects.
    Usually input array is a results of OpenCV face detection and output dlib regtangles are used for landmark detection.

    :param rects_array: array with results of OpenCV face detection
    :return: dlib rectangles object
    """
    rects_dlib = dlib.rectangles()
    for (left, top, right, bottom) in rects_array:
        rects_dlib.append(dlib.rectangle(
            int(left),
            int(top),
            int(right),
            int(bottom)))
    return rects_dlib


def find_landmarks(image, predictor, visualise=False, opencv_facedetector=False):
    """
    Function to find face landmarks (coordinates of nose, eyes, mouth etc) with dlib face landmarks predictor.

    :param image: greyscale image which contains face
    :param predictor: dlib object, shape predictor
    :param visualise: flag to draw detected landmarks on top of the face
    :param opencv_facedetector: flag to switch face detection to OpenCV inplace of dlib HOG detector
    :return: dlib shape object with coordinates for 68 facial landmarks
    """
    if opencv_facedetector:
        # Use OpenCV face detection for a really fast but less accurate results
        faces = _detect_face_opencv(image, face_cascade)
        dets = _array_to_dlib_rectangles(faces)
    else:
        # Use dlib face detection for a more precise face detection,
        # but with lower fps rate
        dets = _detect_face_dlib(image)

    try:
        shape = predictor(image, dets[0])
        i = 0
        if visualise:
            while i < shape.num_parts:
                p = shape.part(i)
                cv2.circle(image, (p.x, p.y), 2, (0, 255, 0), 2)
                i += 1
    except:
        shape = None
    return shape


def face_warp(src_face, src_face_lm, dst_face):
    """
    Function takes two faces and landmarks and warp one face around another according to
    the face landmarks.

    :param src_face: grayscale image (np.array of int) of face which will warped around second face
    :param src_face_lm: landmarks for the src_face
    :param dst_face: grayscale image (np.array of int) which will be replaced by src_face.
                     Landmarks to the `dst_face` will be calculated each time from the image.
    :return: image with warped face
    """
    # Helpers
    output_shape = dst_face.shape[:2]  # dimensions of our final image (from webcam eg)

    # Get the landmarks/parts for the face.
    try:
        dst_face_lm = find_landmarks(dst_face, predictor, opencv_facedetector=True)
        src_face_coord = _shape_to_array(src_face_lm)
        dst_face_coord = _shape_to_array(dst_face_lm)
        warp_trans = skimage.transform.PiecewiseAffineTransform()
        warp_trans.estimate(dst_face_coord, src_face_coord)
        warped_face = skimage.transform.warp(src_face, warp_trans, output_shape=output_shape)
    except:
        warped_face = dst_face

    # Merge two images: new warped face and background of dst image
    # through using a mask (pixel level value is 0)
    warped_face = _merge_images(warped_face, dst_face)
    return warped_face
