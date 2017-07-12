import numpy as np
#from ale_python_interface import ALEInterface
#from scipy.misc import imresize
import random
from environment import BaseEnvironment, FramePool,ObservationPool
import gym

#IMG_SIZE_X = 84
#IMG_SIZE_Y = 84
#NR_IMAGES = 4
ACTION_REPEAT = 1
MAX_START_WAIT = 30
#FRAMES_IN_POOL = 2


class CartPole(BaseEnvironment):
    def __init__(self, actor_id, args):
        self.gym = gym.make('CartPole-v0')

        self.legal_actions = np.array([0,1])
        print("actor_id is " + str(actor_id))
        #self.screen_width, self.screen_height = self.ale.getScreenDims()
        #self.lives = self.ale.lives()
        #
        # self.random_start = args.random_start
        # self.single_life_episodes = args.single_life_episodes
        #self.call_on_new_frame = args.visualize

        # Processed historcal frames that will be fed in to the network 
        # (i.e., four 84x84 images)
       # self.observation_pool = ObservationPool(np.zeros((IMG_SIZE_X, IMG_SIZE_Y, NR_IMAGES), dtype=np.uint8))
       # self.rgb_screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
       # self.gray_screen = np.zeros((self.screen_height, self.screen_width,1), dtype=np.uint8)
       # self.frame_pool = FramePool(np.empty((2, self.screen_height,self.screen_width), dtype=np.uint8),
       #                             self.__process_frame_pool)

    def get_legal_actions(self):
        return self.legal_actions

    # def __get_screen_image(self):
    #     """
    #     Get the current frame luminance
    #     :return: the current frame
    #     """
    #     self.ale.getScreenGrayscale(self.gray_screen)
    #     if self.call_on_new_frame:
    #         self.ale.getScreenRGB(self.rgb_screen)
    #         self.on_new_frame(self.rgb_screen)
    #     return np.squeeze(self.gray_screen)
    #
    # def on_new_frame(self, frame):
    #     pass

    def __new_game(self):
        """ Restart game """
        self.gym.reset_game()


    def __action_repeat(self, a, times=ACTION_REPEAT):
        """ Repeat action and grab screen into frame pool """
        reward = 0
        for i in range(times):
            s_,r,done,info = self.gym.step(self.legal_actions[a])
            reward += r
        return s_,reward,done,info

    def get_initial_state(self):
        """ Get the initial state """
        self.gym.reset()
        observation = self.gym.step(self.gym.action_space.sample())
        return observation[0]

    def next(self, action):
        """ Get the next state, reward, and game over signal """
        observation,reward,terminal,_ = self.__action_repeat(np.argmax(action))
        return observation, reward, terminal


    def get_noop(self):
        return [1.0, 0.0]
