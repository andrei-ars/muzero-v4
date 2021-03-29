# based on gridworld.py and tictactoe.py

import datetime
import os
import sys
import logging

#import gym
import numpy as np
import torch
from gym.utils import seeding
#from webdriverwrapper import Chrome
#from selenium import webdriver

try:
    from .abstract_game import AbstractGame
except:
    from abstract_game import AbstractGame

#import gym
#from gym import spaces
#from gym.utils import seeding, EzPickle

from .enviroment import TestDriverEnviroment 


NUMBER_ACTIONS = 7
MAX_STEPS = 100

WIN_REWARD = 10
POS_REWARD = 0
NEG_REWARD = 0



class MuZeroConfig:
    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (1, 6, 5)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = [i for i in range(NUMBER_ACTIONS)]  # Fixed list of all possible actions. You should only edit the length
        self.players = [i for i in range(1)]  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 1 # 4  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = MAX_STEPS #15  # Maximum number of moves if game is not finished before
        self.num_simulations = 20  # Number of future moves self-simulated
        #self.discount = 0.997  # Chronological discount of the reward
        self.discount = 0.95  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping temperature to 0 (ie playing according to the max)

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 2  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 2  # Number of channels in policy head
        self.resnet_fc_reward_layers = []  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = []  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = []  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 8
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = [64]  # Define the hidden layers in the value network
        self.fc_policy_layers = [64]  # Define the hidden layers in the policy network

        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 20000 # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 1 # 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = True if torch.cuda.is_available() else False  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.001 # 0.005  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000

        ### Replay Buffer
        self.replay_buffer_size = 5000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = False  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = TestDriverEnviroment()

        #cmd = {
        #    "WAIT":     "Wait 1 sec",
        #    "CHOOSE_FIRST_CLICK":  "Choose the firse clickable element",
        #    "CHOOSE_FIRST_SELECT": "Choose the firse selectable element",
        #    "CHOOSE_FIRST_ENTER":  "Choose the firse enterable element",
        #    "NEXT":     "Go to the next active element",
        #    "CLICK":    "Click on the current element",
        #    "ENTER":    "Enter DATA in the current element",
        #    "SELECT":    "Select the current element",
        #    "HIT":      "Hit the current element",
        #    "VERIFY":   "Verify the current URL",
        #    "CLOSE":    "Close the current page",
        #}
        #self.action_number_to_description = {i: cmd[x] for i, x in enumerate(cmd)}
        #self.action_number_to_cmd = {i: x for i, x in enumerate(cmd)}

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done, _ = self.env.step(action)

        #observation = np.expand_dims(observation, axis=0) # 3-dim. array is required

        return np.array(observation), reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.        

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def close(self):
        """
        Properly close the game.
        """
        pass

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        #input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = self.env.action_number_to_description

        return "{}. {}".format(action_number, actions[action_number])

"""
class Selenium_webdriver:
    def __init__(self, url_address, type="Chrome"):
        from webdriverwrapper import Chrome
        from selenium import webdriver
        options = webdriver.ChromeOptions()
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        self.driver = Chrome(options=options)
        # Open a website
        window_before = self.driver.window_handles[0]
        self.driver.get(url_address)

    def get_site_elements(self):
        return None

    def click(self, current_element):
        #self.driver.find_element_by_xpath()
        element = self.driver.find_element_by_name(current_element)
        # the function find_element_by_name should be implemented
        element.click()

    def enter(self, current_element, data):
        element = self.driver.find_element_by_name(current_element)
        enter_field = self.driver.find_element_by_xpath("//input[@name='{}']".format(element))
        enter_field.clear()
        data = generate_data()
        enter_field.send_keys(data)


class Webdriver_imitation:

    def __init__(self):
        #self.site_elements = {'clickables': ['Sign', 'Currency', 'Skip'], 'selectables': [], 'enterables': ['Your email']}
        #self.site_elements = {'clickables': ['Sign', 'Currency', 'Skip'], 'selectables': [], 'enterables': ['Hello']}
        self.reset()

    def reset(self):
        self.site_elements = {
            'clickables': ['Sign', 'Currency', 'Skip'], 
            'selectables': [], 
            'enterables': ['Hi']
            }

    def get_site_elements(self):
        return self.site_elements

    def click(self, current_element_name):
        #print("clickables:", self.site_elements['clickables'])
        #print("current_element_name:", current_element_name)
        if current_element_name in self.site_elements['clickables']:
            index = self.site_elements['clickables'].index(current_element_name)
            self.site_elements['clickables'][index] = None
        else:
            print("! {} is not in the clickables list".format(current_element_name))
            sys.exit()

    def enter(self, current_element_name, data):
        if current_element_name in self.site_elements['enterables']:
            index = self.site_elements['enterables'].index(current_element_name)
            self.site_elements['enterables'][index] = None
        else:
            print("! {} is not in the enterables list".format(current_element_name))
            sys.exit()

def negative_reward():
    #print("-1")
    return NEG_REWARD

def positive_reward():
    #print("+1")
    return POS_REWARD


class TestDriverEnviroment:
    def __init__(self):
        
        self.step_count = 0
        self.total_steps = 0
        self.last_number_of_steps = 0

        self.seed()
        #self.board = numpy.zeros((3, 3)).astype(int)
        # Prepare the webdriver

        #self.driver = Selenium_webdriver(url_address="https://google.com")
        self.driver = Webdriver_imitation()

        # Init the state
        self.chosen_type = None     # 'clickables', 'selectables', 'enterables'
        self.chosen_number = None
        self.reset()

        #cmd = {
            #"CLICK-N": "Click the N-th clickable element",
            #"ENTER-RND": "Enter random text",
            #"OPEN":     "Open the given website",
        #    "WAIT":     "Wait 1 sec",
        #    "NEXT":     "Go to the next active element",
        #    "CHOOSE_FIRST_CLICK":  "Choose the firse clickable element",
        #    "CHOOSE_FIRST_ENTER":  "Choose the firse enterable element",
        #    "CHOOSE_FIRST_SELECT": "Choose the firse selectable element",
        #    "CLICK":    "Click on the current element",
        #    "ENTER":    "Enter DATA in the current element",
        #    "SELECT":    "Select the current element",
        #    "HIT":      "Hit the current element",
        #    "VERIFY":   "Verify the current URL",
        #    "CLOSE":    "Close the current page",
        #}
        #self.action_number_to_description = {i: cmd[x] for i, x in enumerate(cmd)}
        #self.action_number_to_cmd = {i: x for i, x in enumerate(cmd)}

        self.possible_actions = [
            #"CLICK-N": "Click the N-th clickable element",
            #"ENTER-RND": "Enter random text",
            #"OPEN":     "Open the given website",
            (0, "NEXT",     "Go to the next active element"),
            (1, "CLICK",    "Click on the current element"),
            (2, "CHOOSE_FIRST_CLICK",  "Choose the firse clickable element"),
            (3, "ENTER",    "Enter DATA in the current element"),
            (4, "CHOOSE_FIRST_ENTER",  "Choose the firse enterable element"),
            (5, "SELECT",    "Select the current element"),
            (6, "CHOOSE_FIRST_SELECT", "Choose the firse selectable element"),
            #(7, "HIT",      "Hit the current element"),
            #(8, "VERIFY",   "Verify the current URL"),
            #(9, "CLOSE",    "Close the current page"),
            #(10, "WAIT",     "Wait 1 sec"),
        ]
        self.possible_actions = self.possible_actions[:NUMBER_ACTIONS]
        self.cmd_to_element_type = {
            "CHOOSE_FIRST_CLICK": "clickables",
            "CHOOSE_FIRST_SELECT": "selectables",
            "CHOOSE_FIRST_ENTER": "enterables",
            "CLICK": "clickables",
            "SELECT": "selectables",
            "ENTER": "enterables"
        }        

        self.action_number_to_cmd = {i: x[1] for i, x in enumerate(self.possible_actions)}
        self.action_number_to_description = {i: x[2] for i, x in enumerate(self.possible_actions)}

        self.wins = 0
        self.losses = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #def to_play(self):
    #    return 0 if self.player == 1 else 1

    def reset(self):
        #self.board = numpy.zeros((3, 3)).astype(int)
        #self.state = [0]
        self.last_number_of_steps = self.step_count
        self.step_count = 0
        self.driver.reset()
        return self.get_observation()

    def step(self, action):
        self.step_count += 1

        # Call the webdriver and perform the action
        if type(action) is str:
            cmd = action
        else:
            action = int(action)
            cmd = self.action_number_to_cmd[action]
            #print("action type:", type(action))
            #raise Exception("Wrong the action type")
            #sys.exit()

        print("cmd:", cmd)
        #cmd = self.action_number_to_cmd[action]
        reward = 0
        site_elements = self.driver.get_site_elements()
        current_element = None

        if cmd == "WAIT":
            pass

        elif cmd in {"CHOOSE_FIRST_CLICK", "CHOOSE_FIRST_SELECT", "CHOOSE_FIRST_ENTER"}:
            self.chosen_type = self.cmd_to_element_type[cmd]
            if len(site_elements[self.chosen_type]) > 0:
                #current_element = site_elements[self.chosen_type][0]
                self.chosen_number = 0
            else:
                reward = negative_reward()

        elif cmd == "NEXT":
            if self.chosen_type:
                if len(site_elements[self.chosen_type]) > self.chosen_number + 1:
                    self.chosen_number += 1
                    #current_element = site_elements[self.chosen_type][self.chosen_number]
                else:
                    reward = negative_reward()
            else:
                reward = negative_reward()

        elif cmd in {"CLICK", "ENTER", "SELECT"}:
            if self.chosen_number is None:
                reward = negative_reward()
            elif self.chosen_type != self.cmd_to_element_type[cmd]:
                reward = negative_reward()
            elif (self.chosen_type and self.chosen_number < len(site_elements[self.chosen_type])):
                #reward = positive_reward()
                current_element = site_elements[self.chosen_type][self.chosen_number]
                if current_element is None: # perhaps, the element has been already used
                    reward = negative_reward()  # prevent clicking the same element twice
                else:
                    reward = positive_reward()

                    element_number = current_element
                    #element_number = self.chosen_number

                    if cmd == "CLICK":
                        self.driver.click(element_number)
                    elif cmd == "ENTER":
                        self.driver.enter(element_number, data="Hello world")
                    elif cmd == "SELECT":
                        pass
            else:
                reward = negative_reward()

        done = self.have_winner() or len(self.legal_actions()) == 0

        #reward = 1 if self.have_winner() else 0
        if self.have_winner():
            reward = WIN_REWARD * (1 - 0.9*(self.step_count / MAX_STEPS))
            self.wins += 1
            print("{}-th win in {} steps; reward={:.4f}".format(self.wins, self.step_count, reward))
            self.last_number_of_steps = self.step_count
            self.step_count = 0

        return self.get_observation_float(), reward, done, {}

    def get_observation(self):
        # It should return the current state as a numpy array of the float32 type
        # i.e. whole necessary information from the current webpage
        # including the list of active elements and so on.
        # probably, some additional information.
        # It will be fed to neural network.

        #board_player1 = numpy.where(self.board == 1, 1.0, 0.0)
        #board_player2 = numpy.where(self.board == -1, 1.0, 0.0)
        #board_to_play = numpy.full((3, 3), self.player).astype(float)
        #return numpy.array([board_player1, board_player2, board_to_play])

        #site_elements = {'clickables': ['Sign', 'Currency', 'Skip'], 'selectables': [], 'enterables': ['Your email']}
        site_elements = self.driver.get_site_elements()
        #print("site_elements:", site_elements)
        #clickables  = site_elements.get('clickables')
        #selectables = site_elements.get('selectables')
        #enterables  = site_elements.get('enterables')
        de_type = {0: 'clickables', 1: 'selectables', 2: 'enterables'}
        lengths = [len(site_elements[k]) for k in site_elements]
        width = 5
        env_state = [[1 if i<lengths[j] and site_elements[de_type[j]][i] else 0 for i in range(width)] for j in range(len(lengths))]
        env_state = np.array(env_state, dtype=np.int32)

        int_state = np.zeros((3, width), dtype=np.int32)
        de_type_to_number = {y:x for x,y in de_type.items()}
        if self.chosen_type is not None and self.chosen_number is not None:
            de_type_number = de_type_to_number.get(self.chosen_type)
            int_state[de_type_number, self.chosen_number] = 1
        #self.chosen_type]) > self.chosen_number
        state = np.vstack([env_state, int_state])
        state = np.expand_dims(state, axis=0) # 3-dim. array is required
        #clickables = np.array()
        #state = [clickables, selectables, enterables]
        return state

    def get_observation_float(self):
        return np.array(self.get_observation(), dtype=np.float32)

    def legal_actions(self):
        legal = list(range(len(self.possible_actions)))
        return legal

    def have_winner(self):
        #all_active_elements_have_been_clicked = False
        #if all_active_elements_have_been_clicked:
        #    return True
        #else:
        #    return False

        observation = self.get_observation()
        #sum_obs = np.sum(observation)  # only env_state
        sum_obs = np.sum(observation[:,:3])  # only env_state
        #print("sum_obs={}, wins={}\n{}".format(sum_obs, self.wins, str(observation)))
        print("sum_obs={}, wins={}".format(sum_obs, self.wins))
        if sum_obs < 0.01:
            return True
        else:
            return False

    def render(self):
        print("Display the game observation")
        print(self.get_observation())
"""


if __name__ == "__main__":

    #im = Webdriver_imitation()

    env = TestDriverEnviroment()
    print(env.get_observation())
    print(env.legal_actions())
    # 0 - wait, 1 - next,  2 - 1click, 3 - 1select, 4 - 1enter, 5 - click, 6 - enter, 7 - select
    #env.step("WAIT")
    env.step("CHOOSE_FIRST_CLICK")
    env.step("NEXT")
    env.step("CLICK")
    env.step("CLICK")
    print(env.get_observation())

    #print("\n\nTest game")
    #game = Game()
    #game.step("WAIT")
    #game.step("CHOOSE_FIRST_CLICK")
    #game.step("NEXT")
    #game.step("CLICK")
