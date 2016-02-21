import faceWarp
import cv2
import argparse
import sys

# Video file part
def warp_face_in_video(facial_mask_fn, video_in_fn, video_out_fn, show_video=False):
    """
    Function to process frames in video file and 'replace' first found face by the the face from the still image.

    :param facial_mask_fn: path to the still image with a face
    :param video_in_fn: path to the input video file
    :param video_out_fn: path to the video file which will have 'replaced' face
    :param show_video: bool flag to show window with processed video frames
    """

    facial_mask = cv2.imread(facial_mask_fn)
    facial_mask = cv2.cvtColor(facial_mask, cv2.COLOR_BGR2GRAY)
    facial_mask_lm = faceWarp.find_landmarks(facial_mask, faceWarp.predictor)

    video_in = cv2.VideoCapture(video_in_fn)

    video_out = cv2.VideoWriter(
        filename=video_out_fn,
        fourcc=cv2.cv.CV_FOURCC('m', 'p', '4', 'v'),
        frameSize=(int(video_in.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
                   int(video_in.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))),
        fps=25.0,
        isColor=True)

    total_frames_in = video_in.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    while True:
        ret, frame_in = video_in.read()
        if ret == True:
            curr_frame = video_in.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
            if show_video:
                cv2.imshow('video_in', frame_in)
            else:
                print '{:.2%}\r'.format(curr_frame/total_frames_in),
                sys.stdout.flush()
            frame_out = faceWarp.face_warp(facial_mask, facial_mask_lm, frame_in)
            frame_out = cv2.cvtColor(frame_out, cv2.COLOR_GRAY2BGR)
            video_out.write(frame_out)
            if show_video: cv2.imshow('video_out', frame_out)
            cv2.waitKey(1)
        else:
            video_in.release()
            video_in = None
            video_out.release()
            video_out = None
            cv2.destroyAllWindows()
            break

# Video cam part
def warp_face_from_webcam(facial_mask_fn, video_out_fn):
    """
    Function to read video frames from the web cam, replace first found face by the face from the still image
    and show processed frames in a window. Also all processed frames will be save as a video.

    :param facial_mask_fn: path to the still image with a face
    :param video_out_fn: path to the video file which will have 'replaced' face
    """

    facial_mask = cv2.cvtColor(cv2.imread(facial_mask_fn), cv2.COLOR_BGR2GRAY)
    facial_mask_lm = faceWarp.find_landmarks(facial_mask, faceWarp.predictor)

    cam = cv2.VideoCapture(0)
    frame_size = (420, 240) # downsample size, without downsampling too many frames dropped

    video_out = cv2.VideoWriter(
        filename=video_out_fn,
        fourcc=cv2.cv.CV_FOURCC('m', 'p', '4', 'v'), # works good on OSX, for other OS maybe try other codecs
        frameSize=frame_size,
        fps=25.0,
        isColor=True)

    while True:
        ret, frame_in = cam.read()
        # Downsample frame - otherwise processing is too slow
        frame_in = cv2.resize(frame_in, dsize=frame_size)

        frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
        frame_out = faceWarp.face_warp(facial_mask, facial_mask_lm, frame_in)
        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_GRAY2BGR)
        video_out.write(frame_out)
        faceWarp.draw_str(frame_out, (20, 20), 'ESC: stop recording  Space: stop & save video')
        cv2.imshow('webcam', frame_out)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
        if ch == ord(' '):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Let's parse running arguments and decide what we will do
    parser = argparse.ArgumentParser(description='Warp a still-image face around the other face in a video.')
    parser.add_argument('stillface', metavar='STILLFACE',
                        help='the full path to jpg file with face', default='./face_mask.jpg')
    parser.add_argument('inputvideo', metavar='VIDEOIN',
                        help='the full path to input video file where face will be changed. If "0" is provided \
                        then the web cam will be used', default='0')
    parser.add_argument('outputvideo', metavar='VIDEOOUT',
                        help='the full path to output video file with the new face. If "0" is provided then \
                        process video will be shown on the screen, but not saved.')
    args = parser.parse_args()

    # Check if there is a video file and process it.
    if args.inputvideo != '' and args.inputvideo != '0':
        try:
            print '*** Start processing for file: {} ***'.format(args.inputvideo)
            warp_face_in_video(args.stillface, args.inputvideo, args.outputvideo)
            print '\n*** Done! ***'
        except:
            print '*** Something went wrong. Error: {} ***'.format(sys.exc_info())

    # Otherwise use a webcam
    elif args.inputvideo == '0':
        try:
            print '*** Start webcam to save to file: {} ***'.format(args.outputvideo)
            warp_face_from_webcam(args.stillface, args.outputvideo)
            print '\n*** Done! ***'
        except:
            print '*** Something went wrong. Error: {} ***'.format(sys.exc_info())


