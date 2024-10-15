import numpy as np
import pandas as pd
import cv2
import os
# import dxcam
import time
# import keyboard

import win32api
import win32con
import ctypes


class Action:
    """
        Output action on keyboard and mouse
    """

    def __init__(self):
        self.mapVirtualKey = ctypes.windll.user32.MapVirtualKeyA
        self.map = {'w': 87, 's': 83, 'a': 65, 'd': 68, 'r':82 ,'esc': 27, 'enter': 13}
        # self.map = {'w': 38, 's': 40, 'a': 37, 'd': 39, 'r':82 ,'esc': 27, 'enter': 13}

    def down_key(self, value):
        win32api.keybd_event(self.map[value], self.mapVirtualKey(self.map[value], 0), 0, 0)

    def up_key(self, value):
        win32api.keybd_event(self.map[value], self.mapVirtualKey(self.map[value], 0), win32con.KEYEVENTF_KEYUP, 0)

    def press_key(self, value, internal=0.1):
        self.down_key(value)
        time.sleep(internal)
        self.up_key(value)
        time.sleep(0.01)

    def left_click(self, internal=0.2):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(internal)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(0.01)

    def move_mouse(self, pos, interval=0.1, repeat=5):
        for i in range(repeat):
            win32api.SetCursorPos(pos)
            time.sleep(interval)

    def reset(self):
        for key in self.map.keys():
            self.up_key(key)


class Recorder:
    def __init__(self,
                 fps=10,
                 frame_size=(1280, 720),
                 record_img_only=True
                 ):
        self.video_record_fps = fps
        self.frame_size = frame_size
        # self.recorder = dxcam.create(output_idx=0, output_color="BGR")
        # self.recorder.start(target_fps=self.video_record_fps, video_mode=False)

    def get_frame(self):
        return self.recorder.get_latest_frame()


class HighlightRecognizer:
    def __init__(self,
                 loc,
                 bgr_min=(0, 50, 245),
                 bgr_max=(10, 150, 255),
                 img_size=(1280, 720),
                 thres=0.5,
                 debug=False):
        self.highlight_bgr_min = bgr_min
        self.highlight_bgr_max = bgr_max
        self.highlight_thres = thres
        self.loc = loc
        self.img_size = img_size
        self.thres = thres
        self.debug = debug
        self.counter = 0

    def get_roi(self, img):
        x1 = int(self.loc[0] * self.img_size[0])
        x2 = int(self.loc[2] * self.img_size[0])
        y1 = int(self.loc[1] * self.img_size[1])
        y2 = int(self.loc[3] * self.img_size[1])
        # x1, y1, x2, y2 = self.loc
        return img[y1:y2, x1:x2, :]

    def check_highlight(self, img):
        roi = self.get_roi(img)
        lower_bound = np.array(self.highlight_bgr_min, dtype=np.uint8)
        upper_bound = np.array(self.highlight_bgr_max, dtype=np.uint8)
        mask = cv2.inRange(roi, lower_bound, upper_bound)
        num_highlight_pixels_ratio = cv2.countNonZero(mask) / np.prod(mask.shape)
        if self.debug:
            print(f'if_highlight, {num_highlight_pixels_ratio}')
            cv2.rectangle(img, (self.loc[0], self.loc[1]), (self.loc[2], self.loc[3]), (0, 0, 255), 5)
            cv2.imwrite('highlight_roi_{}_{}.jpg'.format(self.counter, num_highlight_pixels_ratio), img)
            self.counter += 1
        if num_highlight_pixels_ratio < self.highlight_thres:
            return False
        return True


class SpeedRecognizer:
    def __init__(self,
                 speed_rects_xys=([0.8713, 0.8850],[0.8870, 0.8850],[0.9036, 0.8850]),
                 speed_rect_size=(0.0130, 0.0400),
                 img_size=(1280, 720),
                 thres=210):
        self.speed_rects_xys = speed_rects_xys
        self.speed_rect_size = speed_rect_size
        self.digit_loc = [
            [0, 0.125, 0.2, 0.8],
            [0.125, 0.4, 0.04, 0.16],
            [0.6, 0.875, 0.04, 0.16],
            [0.875, 1.0, 0.2, 0.8],
            [0.6, 0.875, 0.84, 0.96],
            [0.125, 0.4, 0.84, 0.96],
            [0.46, 0.54, 0.2, 0.8],
        ] # dim0: (y1, y2, x1, x2), dim1:(上，左上，左下，下，右下，右上，中)
        self.img_size = img_size
        self.thres = thres
        self.debug = False

    def get_distance_area(self, img, rect):
        x1 = int(rect[0] * self.img_size[0])
        x2 = int(rect[2] * self.img_size[0])
        y1 = int(rect[1] * self.img_size[1])
        y2 = int(rect[3] * self.img_size[1])
        return img[y1:y2, x1:x2, :]

    def signal_logic(self, signals):
        if signals == [0, 0, 0, 0, 0, 0, 0]:
            return 0
        elif signals == [1, 1, 1, 1, 1, 1, 0]:
            return 0
        elif signals == [0, 0, 0, 0, 1, 1, 0]:
            return 1
        elif signals == [1, 0, 1, 1, 0, 1, 1]:
            return 2
        elif signals == [1, 0, 0, 1, 1, 1, 1]:
            return 3
        elif signals == [0, 1, 0, 0, 1, 1, 1]:
            return 4
        elif signals == [1, 1, 0, 1, 1, 0, 1]:
            return 5
        elif signals == [1, 1, 1, 1, 1, 0, 1]:
            return 6
        elif signals == [1, 1, 0, 0, 1, 1, 0]:
            return 7
        elif signals == [1, 1, 1, 1, 1, 1, 1]:
            return 8
        elif signals == [1, 1, 0, 1, 1, 1, 1]:
            return 9
        else:
            return None

    def single_number_logic(self, img):
        h, w = img.shape[:2]
        areas = []
        for i in range(7):
            y1 = int(self.digit_loc[i][0] * h)
            y2 = int(self.digit_loc[i][1] * h)
            x1 = int(self.digit_loc[i][2] * w)
            x2 = int(self.digit_loc[i][3] * w)
            area = img[y1:y2, x1:x2, :]
            areas.append(area)

        signals = []
        means = []
        for i, area in enumerate(areas):
            # cv2.imwrite("im_digit_{}.jpg".format(i), area)
            # print(np.mean(area))
            means.append(np.mean(area))
            signals.append(1 if means[i] > self.thres else 0)
        num = self.signal_logic(signals)
        return num, signals, means

    def get_speed(self, img, index=None):
        nums = ''
        for i, rect in enumerate(self.speed_rects_xys):
            x1 = self.speed_rects_xys[i][0]
            x2 = x1 + self.speed_rect_size[0]
            y1 = self.speed_rects_xys[i][1]
            y2 = y1 + self.speed_rect_size[1]
            single_num_img = self.get_distance_area(img, (x1, y1, x2, y2))
            num, signals, means = self.single_number_logic(single_num_img)
            if num is None:
                if self.debug:
                    h, w = single_num_img.shape[:2]
                    for j in range(7):
                        y1 = int(self.digit_loc[j][0] * h)
                        y2 = int(self.digit_loc[j][1] * h)
                        x1 = int(self.digit_loc[j][2] * w)
                        x2 = int(self.digit_loc[j][3] * w)
                        cv2.rectangle(single_num_img, (x1, y1), (x2, y2), ((j * 40) % 255, (j * 80) % 255, 255), 1)
                    print('error, img index:{}, digit idx: {}, signals:{}, means:{}'.format(index, i, signals, means))
                    cv2.imwrite('im_error_signals_{}_{}.jpg'.format(index, i), single_num_img)
                    return None
                else:
                    nums += str(0)
            else:
                nums += str(num)
        return int(nums)


if __name__ == '__main__':
    pass