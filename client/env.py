import time
import ctypes
import win32api
import win32ui
import win32gui
import win32con
import numpy as np
import cv2
from utils import SpeedRecognizer, HighlightRecognizer, Recorder, Action
from collections import deque


class WRCGEnv():
    """
    Environment wrapper for WRCG
    """

    def __init__(self,
                 config,
                 ):
        self.game_window_name = "WRCG"
        self.ratio = 1
        self.repeat_nums = 0
        self.repeat_thres = 10
        self.reward_max_speed = 36
        self.reward_coef = 5
        self.stack_penalty = 20
        self.action_penalty = 0.1
        self.gray_scale = config["gray_scale"]
        self.num_concat_image = config["num_concat_image"]
        self.states = {"image": deque(maxlen=self.num_concat_image)}
        self.action_spaces = ['w', 'wa', 'wd', '']
        self.screen_size = (1920, 1080)
        self.resize_size = config["state"]["image"]["dim"][1:]
        self.fps = config["fps"]

        self.with_speed = True if "speed" in config["state"] else False
        if self.with_speed:
            self.states["speed"] = deque(maxlen=self.num_concat_image)

        # button loc
        self.highlight_loc = [0, 0.4417, 0.0336, 0.5]
        self.highlight_ctr = [self.highlight_loc[0] * 0.5 + self.highlight_loc[2] * 0.5,
                              self.highlight_loc[1] * 0.5 + self.highlight_loc[3] * 0.5]
        self.restart_loc = [0.0141, 0.5264]
        self.restart_confirm_loc = [0.3953, 0.7]
        self.start_loc = [0.0164, 0.2125]

        # tools
        self.action = Action()
        self.recorder = Recorder(fps=self.fps, frame_size=self.screen_size)
        self.speedRecognizer = SpeedRecognizer(img_size=self.screen_size)
        self.highlightRecognizer = HighlightRecognizer(img_size=self.screen_size, loc=self.highlight_loc)

        # switch to game window
        hwnd = win32gui.FindWindow(None, self.game_window_name)
        win32gui.SetForegroundWindow(hwnd)

        time.sleep(1)
        self.pause_game()
        time.sleep(1)

    def img_preprocess(self, img):
        """
        preprocess image to a state
        :param img: raw captured image
        :return: preprocess image
        """
        if self.gray_scale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.resize_size)
        return img

    def get_states(self):
        """
        get complete states.
        :return: states
        """
        states = {"image": np.stack(list(self.states["image"]), axis=0)}
        if self.with_speed:
            states["speed"] = np.stack(list(self.states["speed"]), axis=0)
        return states

    def init_states(self):
        """
        initial states when reset
        """
        self.reset_key()
        self.states["image"].clear()
        if self.with_speed:
            self.states["speed"].clear()

        img = self.get_frame()
        state = self.img_preprocess(img)

        for i in range(self.num_concat_image):
            self.states["image"].append(state)
            if self.with_speed:
                self.states["speed"].append(0)
        self.repeat_nums = 0

    def reset_key(self):
        """
        release all keys
        """
        for key in ['w', 's', 'a', 'd']:
            self.action.up_key(key)

    def reset_car(self):
        """
        reset car by pressing R for 2 secs.
        :return: states
        """
        self.reset_key()

        self.action.press_key('r', 2)
        time.sleep(0.1)

        self.init_states()
        return self.get_states()

    def loc_real(self, loc):
        """
        convert float ratio to a real pixel-like location
        :param loc: float (x, y)
        :return: int (x, y)
        """
        return int(loc[0] * self.screen_size[0]), int(loc[1] * self.screen_size[1])

    def reset_game(self):
        """
        reset game when the game is ended by pressing fixed buttons
        :return: states
        """
        self.reset_key()

        self.action.move_mouse(self.loc_real(self.restart_loc))
        self.action.press_key('enter')
        time.sleep(0.2)

        self.action.move_mouse(self.loc_real(self.restart_confirm_loc))
        self.action.press_key('enter')
        time.sleep(10)

        self.action.move_mouse(self.loc_real(self.start_loc))
        self.action.press_key('enter')

        self.init_states()
        return self.get_states()

    def pause_game(self):
        """
        press esc
        """
        time.sleep(1)
        self.action.press_key('esc', internal=0.1)
        time.sleep(3)

    def get_frame(self):
        """
        capture current frame using winapi
        :return: captured image
        """
        w, h = self.screen_size
        hdesktop = win32gui.GetDesktopWindow()
        hwndDC = win32gui.GetWindowDC(hdesktop)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)
        result = saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
        signedIntsArray = saveBitMap.GetBitmapBits(True)
        im_opencv = np.frombuffer(signedIntsArray, dtype='uint8')
        im_opencv.shape = (h, w, 4)
        im_opencv = cv2.cvtColor(im_opencv, cv2.COLOR_BGRA2BGR)
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hdesktop, hwndDC)
        return im_opencv

    def get_speed(self, img):
        """
        get speed for ori image
        :param img: original captured image
        :return: speed of car
        """
        return self.speedRecognizer.get_speed(img)

    def calc_reward(self, speed):
        """
        reward function
        :param speed: speed of current state
        :return: reward of current state
        """
        return min(speed, self.reward_max_speed) / self.reward_max_speed * self.reward_coef - self.action_penalty

    def game_end_check(self, img):
        """
        check if game is ended
        :param img: original captured image
        :return: end flag
        """
        return self.highlightRecognizer.check_highlight(img)

    def step(self, action):
        # do actions
        action_key = self.action_spaces[action]
        for key in ['w', 's', 'a', 'd']:
            if key in action_key:
                self.action.down_key(key)
        if len(action_key) == 2:
            time.sleep(1 / self.fps / 2)
        else:
            time.sleep(1 / self.fps)
        self.reset_key()
        if len(action_key) == 2:
            time.sleep(1 / self.fps / 2)

        # get current capture
        img_ = self.get_frame()

        # calc speed
        speed_ = self.get_speed(img_)

        # calc reward
        reward = self.calc_reward(speed_)

        # update states
        state_ = self.img_preprocess(img_)

        self.states["image"].append(state_)
        if self.with_speed:
            self.states["speed"].append(speed_)


        # if game end
        self.action.move_mouse(self.loc_real(self.highlight_ctr), repeat=5, interval=0.001)
        end = True if self.game_end_check(img_) else False

        # if car stack (done)
        if speed_ == 0:
            self.repeat_nums += 1
        else:
            self.repeat_nums = 0
        done = True if self.repeat_nums >= self.repeat_thres else False
        if done:
            reward -= self.stack_penalty

        if end:
            self.reset_game()

        return {
            "state_prime": self.get_states(),
            "reward": reward,
            "done": done or end
            }


