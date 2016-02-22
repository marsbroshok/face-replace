# Face Replacement Filter in Video
This is a proof-of-concept Python script inspired by [MSQRD](http://msqrd.me) and [Snapchat](https://www.snapchat.com) 3D Face Placement filters.

To play with the script you should have a still image with sample face and video file with someone's else face in it (or you can use your web camera with your own face;) ).

You can find sample still image and video in the "*demo*" directory. 

## Basic Idea

1. Detected where is a face in the still image
2. Extract face features' landmarks (like eyes, nose and mouth coordinates)
3. Detect a face in the video frame and extract its landmarks too
4. Align landmarks coordinates from steps 2 and 3
5. Use results of affine transform to warp still-image face around video frame's face

Final result could be saved to the video file or just shown on the screen.

## Example
Original video, no face replaced:

<a href="https://youtu.be/YqC5wShZCXQ"><img src="https://j.gifs.com/rkE0JW.gif"></a>

Final video, Arni's face is in place of mine:

<a href="https://youtu.be/6X2vD8vt1t4"><img src="https://j.gifs.com/ERZlA4.gif"></a>

## Technical Notes
* All image processing is done on the greyscale frames. If input video is in color, it will be converted to greyscale.
* Web cam and Video in-out were tested on OS X thus video saved with QuickTime codec. 
* To make processing faster all input frames are down-sampled to the size 420x240 px.

## Requirements

This script was tested in the following environment:

* Python 2.7
* [OpenCV 2.4.10](http://opencv.org)
* Skimage
* Numpy
* [Dlib library 18.17](http://dlib.net)
* [Dlib face shape predictor](http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2) (68 face landmarks)


## How To
Before you begin, download and extract [Dlib face shape predictor](http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2) (*'shape_predictor_68_face_landmarks.dat'*). Put it to *./models/shape_predictor_68_face_landmarks.dat*
 
Usage:

`python ./change-face-in-video.py [-h] STILLFACE VIDEOIN VIDEOOUT`

where arguments are:

```
  STILLFACE   the full path to jpg file with face
  VIDEOIN     the full path to input video file where face will be changed. If
              "0" is provided then the web cam will be used
  VIDEOOUT    the full path to output video file with the new face. If "0" is
              provided then process video will be shown on the screen, but not
              saved.
```

For example this command will created a new video `demo_arni.mov` with face replacement from image `arni.jpg`:

`python ./change-face-in-video.py ./demo/arni.jpg ./demo/demo_orig.mov ./demo/demo_arni.mov`

This command will use image `arni.jpg` again, but for the video stream from web camera:

`python ./change-face-in-video.py ./demo/arni.jpg 0 ./demo/demo_arni.mov`

## Credits

[One Millisecond Face Alignment with an Ensemble of Regression Trees by Vahid Kazemi and Josephine Sullivan, CVPR 2014](http://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Kazemi_One_Millisecond_Face_2014_CVPR_paper.html)


-------
Alexander Usoltsev, 2016
