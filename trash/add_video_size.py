import cPickle
import cv2

loadfile = open('D:/Master Thesis/Dataset/landmarks_old.pkl', 'rb')

landmarks = cPickle.load(loadfile)

for person, videos in landmarks.iteritems():
    for video, data in sorted(videos.iteritems()):
        next_frame = 0
        continuous = 0
        num_frames = data['num_frames']
        del data['num_frames']
        cap = cv2.VideoCapture('D:/Master Thesis/Dataset/{}'.format(video))
        data['meta'] = {
            'num_frames': num_frames,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}
        print(data['meta'])
        if not cap.isOpened():
            print('error opening D:/Master Thesis/Dataset/{}'.format(video))
        print 'num_frames:{}'.format(num_frames)

dumpfile = open('D:/Master Thesis/Dataset/landmarks.pkl', 'wb')

cPickle.dump(landmarks, dumpfile, -1)
