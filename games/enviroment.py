# based on gridworld.py and tictactoe.py

import datetime
import os
import sys
import logging
from collections import deque

import numpy as np
import torch
from gym.utils import seeding
#from webdriverwrapper import Chrome
#from selenium import webdriver
from .webdriver_imitation import WebDriverImitaion
#from webdriver_selenium import SeleniumWebDriver


NUMBER_ACTIONS = 7
MAX_STEPS = 100

WIN_REWARD, POS_REWARD, NEG_REWARD = 100, 1, 0 # 100, 1, 0


def negative_reward():
    #print("-1")
    return NEG_REWARD

def positive_reward():
    #print("+1")
    return POS_REWARD


class TestDriverEnviroment:
    """This is enviroment for a test driver
    """
    def __init__(self):

        self.solved_task = False

        self.env_size = (3, 5)
        self.step_count = 0
        self.total_steps = 0
        self.last_num_steps = 0
        self.history_steps = deque()
        self.seed()
        #self.board = numpy.zeros((3, 3)).astype(int)
        # Prepare the webdriver
        self.driver = WebDriverImitaion()
        #self.driver = SeleniumWebDriver(init_url_address="https://demo1.testgold.dev/login")

        # Init the state
        self.chosen_type = None     # 'clickables', 'selectables', 'enterables'
        self.chosen_number = None
        self.reset()

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
        self.last_num_steps = self.step_count
        self.step_count = 0
        self.driver.reset()
        self.prv_state = None
        self.step_history = {'actions': [], 'commands': []}

        return self.get_observation()

    def step(self, action):
        """
        action (int or str)
        """
        self.step_count += 1
        self.total_steps += 1

        # Call the webdriver and perform the action
        if type(action) is str:
            cmd = action
        else:
            action = int(action)
            cmd = self.action_number_to_cmd[action]
            #print("action type:", type(action))
            #raise Exception("Wrong the action type")

        print("wins={}, cmd={}".format(self.wins, cmd))
        #print(cmd)

        #print("{}: wins={} (in {}), obs={} cmd={}".format(
        #    self.step_count, self.wins, self.last_num_steps, self.get_observation()[:15], cmd))
        
        self.step_history['actions'].append(action)

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

                self.step_history['commands'].append((cmd,))

            else:
                reward = negative_reward()

        elif cmd == "NEXT":
            if self.chosen_number is None:
                reward = negative_reward()
            elif self.chosen_type:
                if len(site_elements[self.chosen_type]) > self.chosen_number + 1:
                    self.chosen_number += 1
                    print("self.chosen_number:", self.chosen_number)

                    self.step_history['commands'].append((cmd,))

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
                    
                    #print("current_element:", current_element)

                    element_number = self.chosen_number
                    #element_number = current_element # wrong!

                    if cmd == "CLICK":
                        self.driver.action_on_element("CLICK", element_number)
                        #self.driver.click(current_element)
                    elif cmd == "ENTER":
                        self.driver.action_on_element("ENTER", element_number, data="Hello world")
                        #self.driver.enter(current_element, data="Hello world")
                    elif cmd == "SELECT":
                        pass
                    # it works only if I put current_element instead of self.chosen_number

                    self.step_history['commands'].append((cmd, element_number))

                    if self.solved_task:
                        print("{} {}".format(cmd, self.chosen_number))

            else:
                reward = negative_reward()

        done = self.have_winner() or len(self.legal_actions()) == 0

        #reward = 1 if self.have_winner() else 0
        if self.have_winner():
            reward = WIN_REWARD * (1 - 0.9*(self.step_count / MAX_STEPS))
            self.wins += 1
            
            self.history_steps.append(self.step_count)
            if len(self.history_steps) > 100:
                self.history_steps.popleft()
            avg_steps = np.mean(self.history_steps)
            
            print("{}-th win in {} steps [{:.1f}]; reward={:.4f}".format(
                self.wins, self.step_count, avg_steps, reward))
            self.last_num_steps = self.step_count
            self.step_count = 0

            if avg_steps < 19.0: 
                self.solved_task = True

            print("actions:", self.step_history['actions'])
            print("commands:", self.step_history['commands'])



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
        de_type = {0: 'clickables', 1: 'selectables', 2: 'enterables'}
        lengths = [len(site_elements[k]) for k in site_elements]

        (hight, width) = self.env_size
        # env_state shows which elements have been touched
        env_state = [[1 if i<lengths[j] and site_elements[de_type[j]][i] else 0 for i in range(width)] for j in range(len(lengths))]
        env_state = np.array(env_state, dtype=np.int32)
        
        # int_state shows which element is chosen right now
        int_state = np.zeros((hight, width), dtype=np.int32)
        de_type_to_number = {y:x for x,y in de_type.items()}
        if self.chosen_type is not None and self.chosen_number is not None:
            de_type_number = de_type_to_number.get(self.chosen_type)
            int_state[de_type_number, self.chosen_number] = 1

        if self.prv_state is None:
            self.prv_state = env_state

        #state = np.vstack([env_state, self.prv_state, int_state])
        
        state = np.vstack([env_state, int_state])
        state = np.expand_dims(state, axis=0) # 3-dim. array is required

        #print(state)
        
        #state = state.flatten()
        self.prv_state = env_state

        return state

    def get_observation_float(self):
        return np.array(self.get_observation(), dtype=np.float32)

    def legal_actions(self):
        legal = list(range(len(self.possible_actions)))
        return legal

    def have_winner(self):
        #observation = self.get_observation()
        #sum_obs = np.sum(observation[:self.env_size[0]*self.env_size[1]])
        #return True if sum_obs < 0.01 else False # if all_active_elements_have_been_clicked
        return self.driver.is_target_achieved()

    def render(self):
        print("Display the game observation")
        print(self.get_observation())

    def obs_size(self):
        return 3 * self.env_size[0] * self.env_size[1]

    def number_of_actions(self):
        return NUMBER_ACTIONS


if __name__ == "__main__":

    #im = Webdriver_imitation()

    env = TestDriverEnviroment()
    print(env.get_observation())
    print(env.obs_size())
    print(env.legal_actions())
    env.step("CHOOSE_FIRST_CLICK")
    env.step("NEXT")
    env.step("CLICK")
    env.step("CLICK")
    print(env.get_observation())

