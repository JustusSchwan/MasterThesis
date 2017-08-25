from __future__ import print_function

import glob
from os import path

import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from buildingblocks import landmark_tracker
from dataflow import connectables
from dataflow import logging
from dataflow import ports
from dataflow.connectables import NoneIfNone
from operators import console_hacks
from operators import dlib_tracker
from operators import expressivity
from operators import face_detector
from operators import geometric_features
from operators import utility_positional
from operators import video_source
from operators import viewers
from operators.filter import ButterFilter
from util.np_buffers import AutoCircularArray


def execute_main(dataset_folder, log_folder, logging_mode, activate_view=False):
    # Logging
    logger = logging.Logger(logging_mode)

    # Triggering
    events = ports.TriggerHandler()
    cleanup = ports.TriggerHandler()

    resetter = ports.TriggerHandler()

    # Video Source
    # source = video_source.CameraSource()

    filename_buffer = ports.StateBuffer()
    filename_provider = connectables.IterableStateSource(
         path.basename(p) for p in glob.glob(path.join(dataset_folder, '*.wmv')))
    filename_buffer << filename_provider
    logger.sink_state('filename') << filename_buffer

    source = video_source.FileSource()
    source.sink_filename << \
        NoneIfNone(lambda filename: path.join(dataset_folder, filename)) << \
        logger.source_state('filename')
    source.source_eof >> filename_buffer.reset
    if logger.mode == logging.LoggingMode.RECORD:
        source.source_eof >> (lambda: None if filename_buffer() is None else
                              logger.open('D:/' + path.splitext(path.basename(filename_buffer()))[0] + '.pkl'))

    image_buffer = ports.StateBuffer()
    cleanup.attach(image_buffer.reset)
    image_buffer << source.source_image
    logger.sink_state('image_size') << NoneIfNone(lambda img: img.shape) << image_buffer
    bw_image_buffer = ports.StateBuffer()
    cleanup.attach(bw_image_buffer.reset)
    bw_image_buffer << NoneIfNone(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) << image_buffer

    # Reset handling when person changes
    source.source_file_change >> logger.sink_event('file_change')
    logger.source_event('file_change') >> (lambda x: resetter.trigger())

    # Facial Landmark Detection/Face Tracking
    landmarks = landmark_tracker.LandmarkTrackerBB(image_getter=bw_image_buffer, logger=logger)
    resetter.attach(landmarks.reset_state)

    landmarks_poller = connectables.StateToEvent()
    events.attach(landmarks_poller.get_and_fire)
    landmarks_poller.arg(0) << logger.source_state('landmarks')

    landmarks_poller.source >> geometric_features.get_hand_crafted_geometric_features

    bounds_poller = connectables.StateToEvent()
    events.attach(bounds_poller.get_and_fire)
    bounds_poller.arg(0) << logger.source_state('face_bounds')

    img_size_poller = connectables.StateToEvent()
    events.attach(img_size_poller.get_and_fire)
    img_size_poller.arg(0) << logger.source_state('image_size')

    expr_viewer = None
    mouth_viewer = None
    eyes_viewer = None
    xyz_buffer = AutoCircularArray(3, 10)
    if activate_view:
        # Head Position Estimation
        xyz_position = ports.EventTerminal()
        position_args_poll = connectables.StateToEvent(logger.source_state('landmarks'), logger.source_state('image_size'))
        events.attach(position_args_poll.get_and_fire)
        position_args_poll.source >> utility_positional.get_position_by_average >> xyz_position

        # Head Position Expressivity
        xyz_position >> xyz_buffer
        resetter.attach(xyz_buffer.reset_state)

        ######################################################
        #                   Draw Features                    #
        ######################################################
        expr_viewer = viewers.TimeSeriesViewer(1000)
        expr_viewer.sink_time << source.source_current_frame
        mouth_viewer = viewers.TimeSeriesViewer(1000)
        mouth_viewer.sink_time << source.source_current_frame
        eyes_viewer = viewers.TimeSeriesViewer(1000)
        eyes_viewer.sink_time << source.source_current_frame
        expr_feats = ports.EventTerminal()
        xyz_buffer.source_buf >> expressivity.analyze >> expr_feats
        geom_feats = ports.EventTerminal()
        feats_poller = connectables.StateToEvent(logger.source_state('landmarks'))
        events.attach(feats_poller.get_and_fire)
        feats_poller.source >> geometric_features.get_hand_crafted_geometric_features >> geom_feats
        events.attach(lambda: expr_viewer.draw() if source.source_current_frame() is not None and source.source_current_frame() % 10 == 0 else None)
        events.attach(lambda: mouth_viewer.draw() if source.source_current_frame() is not None and source.source_current_frame() % 10 == 0 else None)
        events.attach(lambda: eyes_viewer.draw() if source.source_current_frame() is not None and source.source_current_frame() % 10 == 0 else None)
        resetter.attach(expr_viewer.clear)
        resetter.attach(mouth_viewer.clear)
        resetter.attach(eyes_viewer.clear)


        filter_params = {'n': 1, 'n_filter': 2, 'cutoff': 1}
        expr_feats >> (lambda x: x[0]) >> ButterFilter(**filter_params) >> expr_viewer.in_series('Spatial Extent', color='black')
        expr_feats >> (lambda x: x[1]) >> ButterFilter(**filter_params) >> expr_viewer.in_series('Speed', color='blue')
        expr_feats >> (lambda x: x[2]) >> ButterFilter(**filter_params) >> expr_viewer.in_series('Fluidity', color='green')

        geom_feats >> (lambda x: x[4]) >> ButterFilter(**filter_params) >> mouth_viewer.in_series('Brow Distance', color='black')
        geom_feats >> (lambda x: x[5]) >> ButterFilter(**filter_params) >> mouth_viewer.in_series('Left Eye Open', color='blue')
        geom_feats >> (lambda x: x[6]) >> ButterFilter(**filter_params) >> mouth_viewer.in_series('Right Eye Open', color='green')

        geom_feats >> (lambda x: x[7]) >> ButterFilter(**filter_params) >> eyes_viewer.in_series('Upper Mouth Angle', color='black')
        geom_feats >> (lambda x: x[8]) >> ButterFilter(**filter_params) >> eyes_viewer.in_series('Lower Mouth Angle', color='blue')
        geom_feats >> (lambda x: x[10]) >> ButterFilter(**filter_params) >> eyes_viewer.in_series('Mouth Open', color='green')

        ######################################################
        #                   Draw Face Detection              #
        ######################################################
        viewer = viewers.ImageViewer('face')
        viewer.sink_image << image_buffer
        viewer.sink_size << logger.source_state('image_size')

        bounds_poller.source >> dlib_tracker.dlib_rect_to_array >> viewer.in_rectangle(color=[255, 255, 0])
        landmarks_poller.source >> face_detector.eye_l >> viewer.in_polyline(color=[0, 255, 0], closed=True)
        landmarks_poller.source >> face_detector.eye_r >> viewer.in_polyline(color=[0, 255, 0], closed=True)
        landmarks_poller.source >> face_detector.mouth_in >> viewer.in_polyline(color=[0, 255, 0], closed=True)
        landmarks_poller.source >> face_detector.mouth_out >> viewer.in_polyline(color=[0, 255, 0], closed=True)
        landmarks_poller.source >> face_detector.outline >> viewer.in_polyline(color=[0, 255, 0], closed=False)
        landmarks_poller.source >> face_detector.brow_l >> viewer.in_polyline(color=[0, 255, 0], closed=False)
        landmarks_poller.source >> face_detector.brow_r >> viewer.in_polyline(color=[0, 255, 0], closed=False)
        landmarks_poller.source >> face_detector.nose_above >> viewer.in_polyline(color=[0, 255, 0], closed=False)
        landmarks_poller.source >> face_detector.nose_below >> viewer.in_polyline(color=[0, 255, 0], closed=False)
        xyz_position >> (lambda x: x * logger.source_state('image_size')()[0]) >> viewer.in_circle([255, 255, 0], 3)
        xyz_buffer.source_buf >> (lambda arr: arr[:, 0:2]) \
            >> (lambda x: x * logger.source_state('image_size')()[0]) >> viewer.in_polyline(color=[255, 255, 0], closed=False)

    logfile_glob = glob.glob(path.join(log_folder, '*.pkl'))
    logfile_iter = iter(logfile_glob)

    progress = console_hacks.ProgressBar()
    source.source_file_change >> progress.start
    source.source_eof >> progress.end

    while True:
        if logger.mode == logging.LoggingMode.REPLAY:
            while not logger.read():
                try:
                    logger.open(logfile_iter.next())
                except StopIteration:
                    break
        events.trigger()
        if logger.source_state('landmarks')() is None:
            xyz_buffer.reset_state()
        if activate_view:
            key = viewer.show()
            if key == 27:  # Escape
                break
            elif key == 2555904:  # right arrow
                source.skip()

        n_frames = source.source_total_frames()
        i_frame = source.source_current_frame()
        if n_frames is not None and i_frame is not None:
            progress.progress(i_frame/(n_frames/100))

        logger.write()
        cleanup.trigger()
        if filename_buffer() is None:
            break

    logger.close()

    if activate_view:
        for viewer in [expr_viewer, mouth_viewer, eyes_viewer]:
            for arr in viewer.series_buffer:
                ydata = arr.get_cropped()[:, 1]
                ydata.reshape(-1, 1)[:] = MinMaxScaler().fit_transform(ydata.reshape(-1, 1))
            viewer.draw()

        plt.ioff()
        plt.show()


if __name__ == '__main__':
    execute_main(dataset_folder='C:/Master Thesis/Dataset', log_folder='D:/Master Thesis/Logfiles',
                 logging_mode=logging.LoggingMode.LIVE, activate_view=True)
