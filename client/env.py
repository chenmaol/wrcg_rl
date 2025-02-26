import time
import ctypes
import win32api
import win32ui
import win32gui
import win32con
import numpy as np
import cv2
import torch

from utils import SpeedRecognizer, HighlightRecognizer, Recorder, Action
from collections import deque
from model import LaneNet


class WRCGBaseEnv:
    """
    Environment wrapper for WRCG
    """

    def __init__(self,
                 config,
                 ):
        """
        Initialize the WRCGBaseEnv class with configuration parameters.
        """
        self.game_window_name = "WRCG"
        self.ratio = 1
        self.repeat_nums = 0
        self.repeat_thres = config["repeat_thres"]
        self.reward_max_speed = config["reward_max_speed"]
        self.reward_coef = config["reward_coef"]
        self.stack_penalty = config["stack_penalty"]
        self.action_penalty = config["action_penalty"]
        self.gray_scale = config["gray_scale"]
        self.num_concat_image = config["num_concat_image"]
        self.states = {"image": [], "speed": []}
        self.action_spaces = config['action_spaces']
        self.screen_size = (1920, 1080)
        self.resize_size = config["resize_size"]
        self.fps = config["fps"]

        self.last_time = time.time()

        self.with_speed = config["with_speed"]

        self.run_type = None

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
            img = np.expand_dims(img, axis=-1)
        img = cv2.resize(img, self.resize_size)
        img = np.transpose(img, (2, 0, 1))
        return img

    def get_states(self):
        """
        get complete states.
        :return: states
        """
        states = {"image": self.states["image"][-1]}
        if self.with_speed:
            states["speed"] = np.array(self.states["speed"][-1])
        return states

    def init_states(self):
        """
        initial states when reset
        """
        self.reset_key()
        self.states["image"].clear()
        self.states["speed"].clear()

        img = self.get_frame()

        self.update_states(img)

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

        if np.random.rand() > 0.5 and self.run_type == "train":
            self.action.press_key('r', 2)
        else:
            self.action.press_key('s', 1)
            self.action.press_key('w', 0.4)
            time.sleep(1)
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

    def calc_reward(self):
        """
        reward function
        :param speed: speed of current state
        :return: reward of current state
        """
        raise NotImplemented

    def game_end_check(self, img):
        """
        check if game is ended
        :param img: original captured image
        :return: end flag
        """
        return self.highlightRecognizer.check_highlight(img)

    def act(self, action):
        """
        perform the action
        """
        raise Exception('not implement')

    def step(self, action):
        """
        execute a step in the environment with the given action
        :param action: action to be performed
        :return: dictionary containing state, action, reward, and done flag
        """
        # do action
        self.act(action)

        # get current capture
        img_ = self.get_frame()

        # update states
        self.update_states(img_)

        # calc reward
        reward = self.calc_reward()

        # if game end
        self.action.move_mouse(self.loc_real(self.highlight_ctr), repeat=5, interval=0.001)
        end = True if self.game_end_check(img_) else False

        # if car stack (done)
        if self.states["speed"][-1] == 0:
            self.repeat_nums += 1
        else:
            self.repeat_nums = 0

        done = True if (self.repeat_nums >= self.repeat_thres) or (
                len(self.states["speed"]) > 0 and self.states["speed"][-2] > 30 + self.states["speed"][-1]) else False

        if done and not end:
            reward = -self.stack_penalty

        if end:
            self.reset_game()

        return {
            "state_prime": self.get_states(),
            "action": action if len(np.array(action).shape) != 0 else np.array(action).reshape(1),
            "reward": np.array(reward).reshape(1),
            "done": np.array(done or end).reshape(1)
        }

    def update_states(self, img):
        """
        更新状态，将图像和速度保存到状态中
        :param img: 当前捕获的图像
        """
        speed = self.get_speed(img)  # 获取速度
        image = self.img_preprocess(img)
        self.states["image"].append(image)
        self.states["speed"].append(np.array([speed], dtype=np.uint8))


class WRCGDiscreteEnv(WRCGBaseEnv):
    """
    Environment wrapper for WRCG
    """

    def __init__(self,
                 config,
                 ):
        """
        Initialize the WRCGDiscreteEnv class with configuration parameters.
        """
        super().__init__(config)

    def calc_reward(self, speed):
        """
        reward function
        :param speed: speed of current state
        :return: reward of current state
        """
        return min(speed, self.reward_max_speed) / self.reward_max_speed * self.reward_coef - self.action_penalty

    def act(self, action):
        """
        perform the action in discrete environment
        :param action: action to be performed
        """
        # do actions
        action_key = self.action_spaces[action]

        # wait for last action finished
        while time.time() - self.last_time < 1 / self.fps:
            pass
        self.reset_key()
        self.last_time = time.time()

        # do current action
        for key in ['w', 's', 'a', 'd']:
            if key in action_key:
                self.action.down_key(key)


class WRCGContinuousEnv(WRCGBaseEnv):
    """
    Environment wrapper for WRCG
    """

    def __init__(self,
                 config,
                 ):
        """
        Initialize the WRCGContinuousEnv class with configuration parameters.
        """
        super().__init__(config)

    def calc_reward(self, speed):
        """
        reward function
        :param speed: speed of current state
        :return: reward of current state
        """
        # if speed > self.reward_max_speed:
        #     r = 1 + (self.reward_max_speed - speed) / self.reward_max_speed
        # else:
        r = min(speed / self.reward_max_speed, 1.0)

        # sudden change speed penalty
        # if self.last_speed and self.last_speed - speed > 30:
        #     r -= self.stack_penalty

        return r * self.reward_coef - self.action_penalty

    def act(self, action):
        """
        perform the action in continuous environment
        :param action: action to be performed
        """
        t1 = (action[0] + 1) / 2 / self.fps
        d = 1 if action[1] > 0 else 2
        t2 = abs(action[1]) / self.fps
        self.action.down_key(self.action_spaces[0])
        self.action.down_key(self.action_spaces[d])
        time.sleep(min(t1, t2))
        if t1 >= t2:
            self.action.up_key(self.action_spaces[d])
        else:
            self.action.up_key(self.action_spaces[0])
        time.sleep(max(t1, t2) - min(t1, t2))
        if t1 >= t2:
            self.action.up_key(self.action_spaces[0])
        else:
            self.action.up_key(self.action_spaces[d])
        time.sleep(1 / self.fps - max(t1, t2))


class WRCGLaneEnv(WRCGDiscreteEnv):
    """
    带有车道线检测的WRCG环境
    """

    def __init__(self, config):
        """
        Initialize the WRCGLaneEnv class with configuration parameters and lane detection model.
        """
        super().__init__(config)

        # 初始化lane detection模型
        self.lane_model = LaneNet(
            size=(288, 800),
            pretrained=False,
            backbone='18',
            cls_dim=(201, 18, 4),
            use_aux=False
        )

        state_dict = torch.load("ep049.pth", map_location='cpu')['model']
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        self.lane_model.load_state_dict(compatible_state_dict, strict=False)
        self.lane_model.eval().to("cuda")

    def img_preprocess(self, img):
        """
        preprocess image for lane detection
        :param img: raw captured image
        :return: processed image for lane detection
        """
        if self.gray_scale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)
        img = cv2.resize(img, (800, 288)) / 255.
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0).to("cuda")

        lane = self.lane_model(img)[..., :2].squeeze().detach().cpu().numpy()
        return lane

    def calc_reward(self):
        """
        根据速度和车道线计算奖励
        :return: calculated reward based on speed and lane position
        """
        # 基础速度奖励
        speed_reward = min(np.array(self.states["speed"][-1]),
                           self.reward_max_speed) / self.reward_max_speed * self.reward_coef

        # lane reward
        vehicle_position = (640, 450)
        img_h, img_w = (800, 1280)
        col_sample_w = 4
        lane_line = np.array(self.states["image"][-1])  # shape: (201, 18, 2)
        out_j = np.argmax(lane_line, axis=0)  # shape: (18, 2)
        center_lane = []

        row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
        for i in range(len(out_j)):  # [18, 2]
            if out_j[i, 0] > 0 and out_j[i, 1] > 0:
                center_j = out_j[i, 0] * 0.5 + out_j[i, 1] * 0.5
                ppp = (int(center_j * col_sample_w * img_w / 800) - 1,
                       int(img_h * (row_anchor[len(row_anchor) - 1 - i] / 288)) - 1)
                center_lane.append(ppp)
        center_lane = np.array(center_lane)
        lane_reward = 0
        if len(center_lane) >= 2:
            distances = np.linalg.norm(center_lane - vehicle_position, axis=1)
            closest_point_idx = np.argmin(distances)
            closest_point = center_lane[closest_point_idx]

            # 计算横向偏差（x方向）
            lateral_deviation = (vehicle_position[0] - closest_point[0]) / (img_w / 2) # -1 ~ 1
            lane_reward = 1 - np.exp(-3 * (1 - np.abs(lateral_deviation)))

        return speed_reward + lane_reward * 0.3 - self.action_penalty

    def get_states(self):
        """
        get complete states including lane information.
        :return: states
        """
        states = {"image": self.states["image"][-1].reshape(-1)}
        if self.with_speed:
            states["speed"] = np.array(self.states["speed"][-1])
        return states
