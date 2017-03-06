import cv2
import numpy as np
import os

class pipeline:
    """
    a video pipeline for track finding
    """
    def __init__(self, undistort_eye, depth_eye, tracker, funcs, weight = 0.4):
        self.u_cam = undistort_eye
        self.p_cam = depth_eye
        self.trk = tracker
        self.funcs = funcs
        self.w = weight

    def make(self, img):
        """
        params:
            img: numpy array, the frame
            undistort_camera: a calibrated camera
            perspective_camera: a fitted perspective_camera
            tracker: a lane tracker
            funcs: pipeline function that returns binary mask
            weight: beta

        return: numpy array, video frame with lane drawn
        """
        undist = self.u_cam.cal_undist(img)
        med = undist

        for func in self.funcs:
            med = func(med)

        warped_mask = self.p_cam.transform(med)
        out = self.trk.lane(warped_mask)
        newwarp = self.p_cam.inv_transform(out)
        result = cv2.addWeighted(undist, 1, newwarp, self.w, 0)
        return result
