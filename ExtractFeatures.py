from __future__ import print_function

import glob
import os
import re
from os import path

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from dataflow import connectables
from dataflow import logging
from dataflow import ports
from dataflow.connectables import NoneIfNone
from operators import clustering
from operators import expressivity
from operators import geometric_features
from operators import utility_positional
from operators import video_source
from util.buffers import GrowingArray
from util.np_buffers import AutoCircularArray
from util.np_buffers import GrowingNumpyArray


def main(dataset_folder, log_folder, logging_mode, output_folder=None, agglomerative_n=None, dbscan_n=None, dbscan_eps=None,
         metric='pseudo_cosine'):
    events = ports.TriggerHandler()
    events_cleanup = ports.TriggerHandler()
    events_new_file = ports.TriggerHandler()
    events_file_end = ports.TriggerHandler()
    events_candidate_done = ports.TriggerHandler()
    events_landmarks_lost = ports.TriggerHandler()

    logger = logging.Logger(logging.LoggingMode.REPLAY)
    feature_logger = logging.Logger(logging_mode)

    # Timer
    frame_counter = connectables.Counter()
    events_new_file.attach(frame_counter.reset)
    current_time = ports.StateTerminal()
    framerate = 10
    current_time << (lambda frame: float(frame) / framerate) << frame_counter

    # Candidate/session name
    filename_buffer = ports.StateBuffer()
    events_new_file.attach(filename_buffer.reset)
    events_new_file.attach(filename_buffer)
    filename_buffer << logger.source_state('filename')
    name_regex = re.compile('(.+)_(\\d+)\\.wmv')
    name_source = ports.StateTerminal()
    session_source = ports.StateTerminal()
    name_source << (lambda string: name_regex.match(string).expand('\\1')) << filename_buffer
    session_source << (lambda string: name_regex.match(string).expand('\\2')) << filename_buffer

    previous_name_source = ports.StateBuffer()
    previous_name_source << name_source
    events_new_file.attach(previous_name_source)

    # Set up triggering of new candidate
    name_poller = connectables.StateToEvent()
    events_new_file.attach(name_poller.get_and_fire)
    new_candidate = connectables.EventEdge()
    name_poller.arg(0) << name_source
    name_poller.source >> new_candidate
    new_candidate.source >> events_candidate_done.trigger
    new_candidate.source >> print

    # Expressivity Features
    pos_buffer = AutoCircularArray(3, 20)
    events_landmarks_lost.attach(pos_buffer.reset_state)
    events_new_file.attach(pos_buffer.reset_state)
    pos_buffer_poll = connectables.StateArgumentZip(logger.source_state('landmarks'), logger.source_state('image_size'))
    xy_expressivity = ports.StateTerminal()
    pos_buffer_poll << NoneIfNone(utility_positional.get_position_by_average)
    xy_expressivity << NoneIfNone(expressivity.analyze) << NoneIfNone(pos_buffer) << pos_buffer_poll

    # Geometric Features
    geometric_feats = ports.StateTerminal()
    geometric_feats << NoneIfNone(geometric_features.get_hand_crafted_geometric_features) << logger.source_state(
        'landmarks')

    # Concatenate all features to feature vector
    features_catted = connectables.StateArgumentZip(xy_expressivity, geometric_feats)
    features_catted << NoneIfNone(lambda *args: np.concatenate(args, axis=0))
    feature_logger.sink_state('features') << features_catted
    feature_vector = feature_logger.source_state('features')

    if output_folder is not None:
        # Store Video and frame information
        file_frame_store = GrowingArray()
        face_bounds_store = GrowingArray()

        # Concatenate feature vectors into feature matrix
        feature_matrix_store = GrowingNumpyArray()
        all_feature_matrix = GrowingNumpyArray()
        feature_poller = connectables.StateToEvent()
        events.attach(feature_poller.get_and_fire)
        feature_poller.arg(0) << feature_vector
        feature_poller.source >> feature_matrix_store.append
        feature_poller.source >> all_feature_matrix.append
        feature_poller.source >> NoneIfNone(lambda _: file_frame_store.append((path.join(dataset_folder, filename_buffer()),
                                                                               frame_counter())))
        feature_poller.source >> NoneIfNone(lambda _: face_bounds_store.append(logger.source_state('face_bounds')()))
        feature_matrix = ports.StateBuffer()

        if metric == 'euclidean':
            feature_matrix << \
                NoneIfNone(lambda mat: StandardScaler().fit_transform(mat)) << \
                feature_matrix_store
        elif metric == 'pseudo_cosine':
            feature_matrix << \
                NoneIfNone(lambda mat: normalize(mat)) << \
                NoneIfNone(lambda mat: MinMaxScaler().fit_transform(mat)) << \
                feature_matrix_store
        else:
            raise RuntimeError("You must specify 'euclidean' or 'pseudo cosine' as distance metric")

        # Clustering
        cluster_algo = None
        if agglomerative_n is not None:
            cluster_algo = clustering.SimpleAgglometrative(n_clusters=agglomerative_n, linkage='ward', affinity='euclidean')
        else:
            assert dbscan_eps is not None and dbscan_n is not None
            cluster_algo = clustering.SimpleDBSCAN(dist=dbscan_eps, min_neighborhood=dbscan_n)

        cluster_algo.sink_data << feature_matrix
        cluster_result = ports.StateBuffer()
        cluster_result << cluster_algo.do_clustering
        cluster_metric = clustering.SilhouetteScore()
        cluster_metric.sink_data << feature_matrix
        cluster_result_poller = connectables.StateToEvent()
        cluster_result_poller.arg(0) << cluster_result
        cluster_result_poller.source >> (lambda cl: set(cl.labels_)) >> print

        frame_source = video_source.FrameSource()

        def crop_face(num):
            img = frame_source.source_frame(*file_frame_store[num])
            if img is None:
                return None
            rect = face_bounds_store[num]
            if rect is None:
                return None
            return img[int(rect.top()):int(rect.bottom()), int(rect.left()):int(rect.right())]

        # Plot Clusters
        result_folder_name = ''
        if agglomerative_n is not None:
            result_folder_name = path.join(output_folder,
                                           'clustering_results_agglomerative_{}_{}'.format(metric, agglomerative_n))
        else:
            assert dbscan_eps is not None and dbscan_n is not None
            result_folder_name = path.join(output_folder,
                                           'clustering_results_dbscan_{}_{}_eps_{}'.format(metric, dbscan_n,
                                                                                           str(dbscan_eps).replace('.',
                                                                                                                   '_')))
        if not path.exists(result_folder_name):
            os.mkdir(result_folder_name)
        cluster_printer = clustering.ClusterPrinter(num_images=100)
        cluster_printer.sink_features << feature_matrix
        cluster_printer.sink_image << (lambda feature_num: crop_face(feature_num))
        cluster_printer.sink_filename << (
            lambda: path.join(result_folder_name, previous_name_source() + '.png'))
        cluster_result_poller.source >> cluster_printer
        cluster_result_poller.source >> cluster_metric \
            >> connectables.Tee(print) \
            >> (lambda score: os.rename(path.join(result_folder_name, previous_name_source() + '.png'),
                                        path.join(result_folder_name, previous_name_source() + '_'
                                                  + '{}'.format(score) + '.png')))

        cluster_grid_printer = clustering.ClusterSamplePrinter(num_images=20, num_rows=1)
        cluster_grid_printer.sink_image << (lambda feature_num: crop_face(feature_num))
        cluster_grid_printer.sink_filename << (
            lambda: path.join(result_folder_name, "sample_" + previous_name_source() + '.png'))
        cluster_result_poller.source >> cluster_grid_printer

        # Clean up clustering
        events_candidate_done.attach(cluster_result_poller.get_and_fire)
        events_candidate_done.attach(cluster_result.reset)

        events_candidate_done.attach(feature_matrix_store.clear)
        events_candidate_done.attach(feature_matrix.reset)
        events_candidate_done.attach(file_frame_store.clear)
        events_candidate_done.attach(face_bounds_store.clear)

    events_candidate_done.attach(previous_name_source.reset)

    logfile_glob = glob.glob(path.join(log_folder, '*.pkl'))
    logfile_iter = iter(logfile_glob)
    first_file = True
    while True:
        opened = False
        finalized = False
        if logger.mode == logging.LoggingMode.REPLAY:
            while not logger.read():
                try:
                    filename = logfile_iter.next()
                    if logger.open(filename):
                        opened = True
                        feature_logger.open(path.splitext(filename)[0] + '.features')
                        # feature_logger.open(path.join(tempfile.gettempdir(), 'tmp.feature'))
                    else:
                        continue
                except StopIteration:
                    finalized = True
                    break
            feature_logger.read()
        if finalized:
            events_file_end.trigger()
            events_candidate_done.trigger()
            break
        if opened:
            if not first_file:
                events_file_end.trigger()
            first_file = False
            events_new_file.trigger()
            events_cleanup.trigger()

        if logger.source_state('landmarks')() is None:
            events_landmarks_lost.trigger()

        events.trigger()

        frame_counter.advance()

        # if filename_buffer() is None:
        #     break

        logger.write()
        feature_logger.write()

        events_cleanup.trigger()

    logger.close()
    feature_logger.close()


if __name__ == '__main__':
    # main(10)
    for c in [8]:
        # main(dbscan_eps=0.15, dbscan_n=10)
        main(dataset_folder='C:/Master Thesis/Dataset', log_folder='D:/Master Thesis/Logfiles',
             logging_mode=logging.LoggingMode.LIVE,
             output_folder='D:/',
             agglomerative_n=c, metric='pseudo_cosine')
