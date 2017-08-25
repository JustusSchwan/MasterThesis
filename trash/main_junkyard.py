# Estimate Skin Color

skin = skin_color.SkinColorEstimator()
skin.sink_image << image_buffer
skin.sink_landmarks << landmarks.landmarks_buffer
skin_mask_buffer = ports.StateBuffer()
cleanup.attach(skin_mask_buffer.reset_state)
skin_mask_buffer.sink << skin.get_skin_color_mask

# close_mask = circle_mask(2)
# open_mask = circle_mask(1)
# >> (lambda img: cv2.threshold(img, thresh=50, maxval=255, type=cv2.THRESH_BINARY)[1]) \
# >> (lambda img: cv2.morphologyEx(img, kernel=close_mask, op=cv2.MORPH_CLOSE)) \
# >> (lambda img: cv2.morphologyEx(img, kernel=open_mask, op=cv2.MORPH_OPEN)) \

# Cluster Skin colored regions
clustering_downsample = 2
cluster_dist = 4
# skin_clusters = clustering.BlobClusteringDBSCAN(dist=cluster_dist, min_neighborhood=np.power(cluster_dist, 2) * 0.8,
# thresh_intensity=25, scale_intensity=cluster_dist * 2 / 255, debug=True)

skin_clusters = clustering.BlobClusteringConnectivity(thresh_intensity=25, debug=False)


def subtract_face_area(img):
    if img is None:
        return None
    ltrb = np.array(dlib_tracker.dlib_rect_to_array(landmarks.face_bounds_buffer()), dtype=np.int32)
    h = ltrb[3]-ltrb[1]
    miny = max(ltrb[1] - h/2, 0)
    maxy = min(ltrb[3] + h/2, img_size_buffer()[0])
    img[miny:maxy, ltrb[0]:ltrb[2]] = 0
    return img


skin_clusters.sink_image << \
    (lambda img: None if img is None else cv2.resize(img, dsize=None, fx=1.0 / clustering_downsample,
                                                     fy=1.0 / clustering_downsample, interpolation=cv2.INTER_AREA)) << \
    subtract_face_area << \
    (lambda img: None if img is None else cv2.medianBlur(img, 5)) << \
    skin_mask_buffer

skin_clusters_buffer = ports.StateBuffer()
cleanup.attach(skin_clusters_buffer.reset_state)
skin_clusters_buffer.sink << \
    clustering.cluster_filter_op(condition_counts=lambda c: c > 500) << \
    clustering.cluster_scaler(clustering_downsample) << \
    skin_clusters.make_clusters

skin_clusters.out_debug_image \
    >> (lambda img: cv2.resize(img, dsize=None, fx=clustering_downsample, fy=clustering_downsample)) \
    >> viewer.in_image

# Hand detection
hand_l = hand_tracker.HandTracker(left=True)
hand_l.sink_image << image_buffer
hand_l.sink_skin_color << skin_mask_buffer
hand_l.sink_face_bounds << landmarks.face_bounds_buffer
hand_l.sink_clusters << skin_clusters_buffer

hand_r = hand_tracker.HandTracker(left=False)
hand_r.sink_image << image_buffer
hand_r.sink_skin_color << skin_mask_buffer
hand_r.sink_face_bounds << landmarks.face_bounds_buffer
hand_r.sink_clusters << skin_clusters_buffer








