import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


RED, GREEN, BLUE = (255*np.eye(3,dtype = int)).tolist()


class tracker:
    """
    the lane finder for a video(or series of images in same size)
    serves as a part of the pipeline
    """
    def __init__(self, margin, nb_win, minpix, mpp_y, frac = 4, curv_count = 200, lane_width = 3.7):
        """
        params:
            margin: int, the half width of the sliding window
            nb_win: int, times sliding
            minpix: int, minimun pixel to be recognized as a successful search
            mpp_y: float, meters per pixel in the y direction
            frac: int, 1/frac of the mask will be used to calculate the histogram
            curv_count: int, the fit will only be updated when at least curv_count pixels is above 1/frac line of the image.
                        The lane close to the driver is easy to see so in some case the algorithm will overfit the lower part of the mask.
            lane_width: float, in meters, default is the
        """
        self.half = margin
        self.nb_win = nb_win
        self.minpix = minpix
        self.frac = frac
        self.fit = None
        self.w = None
        self.h = None
        self.lane_width = lane_width
        self.midpoint = None
        self.curv_count = curv_count
        self.mpp_y = mpp_y

    def curvature(self, mpp_x, y_val = None):
        """
        measure the curvature of both lanes found
        reference: http://www.intmath.com/applications-differentiation/8-radius-curvature.php

        params:
            y_val: int, where we want the curvature

        return: tuple of radius for left and right lanes
        """
        l_fit, r_fit = self.fit
        y = y_val if y_val else self.h
        mpp_ratio = mpp_x/self.mpp_y
        radius = lambda y, fit : (1 + mpp_ratio**2*(2*fit[0]*y + fit[1])**2)**1.5/np.abs(2*fit[0]*mpp_ratio**2/mpp_x)
        return radius(y, l_fit), radius(y, r_fit)


    def lane(self, mask, win_color = None):
        """
        get the lane out of a binary mask

        params:
            mask: numpy array, binary mask
            win_color: for test, color for drawing windows

        return:
            numpy array, a video frame
        """

        # the nonzero point
        solid = np.nonzero(mask)
        sx, sy = solid[1], solid[0]

        # make a image to draw on
        out_img = np.dstack([np.zeros_like(mask)]*3)*255
        if self.fit is None:
            #  get the intial poly line for window sliding

            # get the midpoint for both line, expecting it shows up in the lower half
            self.h, self.w = mask.shape
            self.midpoint = self.w//2
            self.win_height = self.h//self.nb_win

            curv_head = self.h//self.frac
            histogram = np.sum(mask[:curv_head, :], axis = 0)
            mid_l = np.argmax(histogram[:self.midpoint])
            mid_r = np.argmax(histogram[self.midpoint:]) + self.midpoint

            # the indice for solid pixel in left and right
            l_lane_idc = []
            r_lane_idc = []

            # slide the windows down up
            btm = self.h
            for n in range(self.nb_win):
                # right window
                ul_l = (mid_l - self.half, btm - self.win_height)
                lr_l = (mid_l + self.half, btm)

                # left window
                ul_r = (mid_r - self.half, btm - self.win_height)
                lr_r = (mid_r + self.half, btm)


                # draw the retangle on the image
                if win_color:
                    cv2.rectangle(out_img, lr_l, ul_l, win_color, 2)
                    cv2.rectangle(out_img, lr_r, ul_r, win_color, 2)


                # the indice within window
                within_l = ((sx>=ul_l[0]) & \
                            (sx<=lr_l[0]) & \
                            (sy>=ul_l[1]) & \
                            (sy<=lr_l[1])).nonzero()[0]

                within_r = ((sx>=ul_r[0]) & \
                            (sx<=lr_r[0]) & \
                            (sy>=ul_r[1]) & \
                            (sy<=lr_r[1])).nonzero()[0]

                # append to the lane
                l_lane_idc.append(within_l)
                r_lane_idc.append(within_r)

                if len(within_r) > self.minpix:
                    mid_r = np.int(np.mean(sx[within_r]))
                if len(within_l) > self.minpix:
                    mid_l = np.int(np.mean(sx[within_l]))
                btm -= self.win_height

            # concatenate the windows
            l_lane_idc = np.concatenate(l_lane_idc)
            r_lane_idc = np.concatenate(r_lane_idc)
            try:
                self.fit = [np.polyfit(sy[l_lane_idc], sx[l_lane_idc], 2),
                            np.polyfit(sy[r_lane_idc], sx[r_lane_idc], 2)]
            except:
                return out_img


        else:
            # if we've fitted the lane, use that as guide
            l_fit, r_fit = self.fit
            l_lane_idc = ((sx >= np.polyval(l_fit, sy) - self.half) &
                          (sx <= np.polyval(l_fit, sy) + self.half)).nonzero()[0]
            r_lane_idc = ((sx >= np.polyval(r_fit, sy) - self.half) &
                          (sx <= np.polyval(r_fit, sy) + self.half)).nonzero()[0]


            curv_head = self.h//self.frac
            l_curv_count = np.sum((sy >= curv_head) & (sx <= self.midpoint))
            r_curv_count = np.sum((sy >= curv_head) & (sx >= self.midpoint))

            if l_curv_count >= self.curv_count:
                try: self.fit[0] = np.polyfit(sy[l_lane_idc], sx[l_lane_idc], 2)
                except: pass
            if r_curv_count >= self.curv_count:
                try: self.fit[1] = np.polyfit(sy[r_lane_idc], sx[r_lane_idc], 2)
                except: pass

        # draw the lane area
        l_fit, r_fit = self.fit
        y_cord = np.linspace(0, self.h - 1, self.h)
        lane_l = np.polyval(l_fit, y_cord)
        lane_r = np.polyval(r_fit, y_cord)


        if not win_color:
            pts_l = np.array([np.vstack([lane_l, y_cord]).T])
            pts_r = np.array([np.flipud(np.vstack([lane_r, y_cord]).T)])

            pts = np.hstack((pts_l, pts_r))
            cv2.fillPoly(out_img, np.int_(pts), [0, 100, 0])

        # draw red on left
        out_img[sy[l_lane_idc], sx[l_lane_idc]] = RED
        # draw blue on right
        out_img[sy[r_lane_idc], sx[r_lane_idc]] = BLUE


        # put text showing meters away center and radius
        l_btm = np.polyval(l_fit, self.h)
        r_btm = np.polyval(r_fit, self.h)
        mpp = self.lane_width/(r_btm - l_btm) # meters per pixel

        mid_lane = int((r_btm + l_btm)/2)
        dev = (self.midpoint - mid_lane)
        radius = np.mean(self.curvature(mpp))

        side = ''
        side = 'L' if dev < 0 else 'R'
        dev_text = ("%.2fm %s"%(np.abs(mpp*dev), side))
        radius_text = ("RADIUS %.2fm"%(radius)) if radius < 2000 else 'STRAIGHT'

        (dev_w, dev_h), _ = cv2.getTextSize(dev_text,
                    fontFace =  cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1, thickness = 2)

        (radius_w, radius_h), _ = cv2.getTextSize(radius_text,
                    fontFace =  cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1, thickness = 3)


        dev_org = (int(mid_lane + 2*dev - dev_w//2), self.h - 30)
        radius_org = (int(mid_lane - radius_w//2), self.h - 80)



        cv2.line(out_img, (mid_lane, self.h - 20),
                          (mid_lane, self.h - 40 - dev_h),
                          color = [255,255,255], thickness = 3)

        cv2.putText(out_img, radius_text,
                    fontFace =  cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1, thickness = 3,
                    org = radius_org, color = [0, 0, 0])

        cv2.putText(out_img, dev_text,
                    fontFace =  cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 1, thickness = 2,
                    org = dev_org, color = [0, 0, 0])

        return out_img





def test():
    mask = np.load("mask.npy")
    trk = tracker(120, 9, 20, 0.054)
    out = trk.lane(mask)
    out = trk.lane(mask)
    out = trk.lane(mask)
    plt.imshow(out)
    plt.show()

#test()
