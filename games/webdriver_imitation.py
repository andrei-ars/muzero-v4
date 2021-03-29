# based on gridworld.py and tictactoe.py
"""
Requirements
pip install lxml
"""

import time
import datetime
import os
import sys
import logging
import numpy as np
import torch
# for Selenium_webdriver
from bs4 import BeautifulSoup
from .xpath_soup import xpath_soup
from .datagen import DataGenerator
from .logger import Logger


class AbstractWebDriver():
    def __init__(self, init_url_address):
        pass
    def __init__(self, init_url_address):
        pass
    def reset(self, initial=False):
        pass
    def get_site_elements(self):
        pass
    def action_on_element(self, action, element_number, data=None):
        pass
    def is_target_achieved(self, targets=None):
        pass


class WebDriverImitaion(AbstractWebDriver):

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_action = None
        self.site_elements = {
            'clickables':  [0],
            'selectables': [0], 
            'enterables':  [0, 0, 0]
            }

    def get_site_elements(self):
        return self.site_elements

    def is_target_achieved(self):
        #observation = self.get_observation()
        #sum_obs = np.sum(observation[:self.env_size[0]*self.env_size[1]])
        #return True if sum_obs < 0.01 else False # if all_active_elements_have_been_clicked
        if self.last_action == "CLICK"\
                and self.site_elements['enterables'][0] == 1\
                and self.site_elements['enterables'][1] == 1\
                and self.site_elements['clickables'][0] > 0:
            return True
        else:
            return False

    def action_on_element(self, action, element_number, data=None):
        #print("{} {}".format(action, element_number))
        self.last_action = action
        action_to_element_type = {"CLICK": "clickables", "ENTER": "enterables", "SELECT": "selectables"}
        element_type = action_to_element_type[action]
        if element_number < len(self.site_elements[element_type]):
            self.site_elements[element_type][element_number] += 1
            return True
        else:
            return False






class WebDriverImitaion_v01(AbstractWebDriver):
    def __init__(self, init_url_address):
        """ Return chrome webdriver and open the initial webpage.
        """
        print("WebDriverImitaion Initialization")
        self.delay_after_click = 0
        self.logger = Logger("_log.log")
        self.log = self.logger.log
        self.init_url_address = init_url_address
        self.data_generator = DataGenerator(self)
        self.reset(initial=True)

    def reset(self, initial=False):
        """ This function doesn't reboot (reopen) browser each time but only if necessary.
        Set initial is True for the first time only.
        """
        self.log("RESET")
        self.right_seq = False
        self.history = []
        self.page_elements = None
        self.is_page_has_been_updated = False
        self.previos_html_code = None
        #current_url = self.init_url_address
        self.log("self.init_url_address: {}".format(self.init_url_address))
        self._clear_page()

    def print_state(self):
        print("!self.page_state:", self.page_state)
        time.sleep(1)

    def get_site_elements(self):
        #logging.debug("get_site_elements")
        #html_code = self._get_html_code()
        #elements = self._get_page_elements(self.html_code)
        self.site_elements = {
            'clickables': ["['btn', 'btn-primary']"], 
            'selectables': [], 
            'enterables': ['email', 'password', 'remember']}
        #print("site_elements: {}".format(self.site_elements))
        return self.site_elements

    def action_on_element(self, action, element_number, data=None):
        """
        """
        #print("action={}, element_number={}".format(action, element_number))
        action_to_element_type = {"CLICK": "button", "ENTER": "input"}
        element_type = action_to_element_type[action]
        #element = self.page_elements[element_type][element_number]
        #driver_element = self.driver.find_element_by_xpath(element.xpath)

        if action == "CLICK":
            #print("Click on the element #{}".format(element_number))
            #driver_element.click()
            self.page_state['clickables'][element_number] += 1
            self.logger.log("CLICK [{}]".format(element_number))
            #time.sleep(self.delay_after_click)
            if self._check_enterables_fields():
                self.right_seq = True
                print("CLICK - self.page_state:", self.page_state)
                #time.sleep(1)

        elif action == "ENTER":
            # driver_element.clear()
            element_name = self.site_elements['enterables'][element_number]
            data = self.data_generator.infer(element_name)
            #print("Enter into the element #{}".format(element_number))
            #driver_element.send_keys(data)
            self.page_state['enterables'][element_number] += data
            element_name = self.get_site_elements()['enterables'][element_number]
            self.logger.log("ENTER [{} ({})]: data=\"{}\"".format(element_number, element_name, data))
        return True

    def is_target_achieved(self, targets=None):
        #return self.is_page_has_been_updated

        current_url = self._get_current_url()
        #print("current_url:", current_url)

        if targets:
            is_achieved = False
            final_targets = targets.get('final')
            for target in final_targets:
                target_url = target.get('url')
                if current_url == target_url:
                    print("current_url is target_url:", target_url)
                    self.logger.log("!current_url is target_url: {}".format(target_url))
                    is_achieved = True
        else:
            elements = self._get_page_elements(self.html_code)
            inputs = elements['input']
            singin_window = False
            for input0 in inputs:
                if input0.name == "password":
                    singin_window = True
            if not singin_window:
                print("The target is achieved. This is not a singin_window")
            is_achieved = not singin_window

        return is_achieved

    def _get_html_code(self):
        self.html_code = self.driver.page_source
        if self.previos_html_code is not None and self.previos_html_code != self.html_code:
            self.is_page_has_been_updated = True
            self.logger.log("page_has_been_updated")
        self.previos_html_code = self.html_code
        return self.html_code


    def _get_current_url(self):
        if self.right_seq:
            url = "https://demo1.testgold.dev/panel"
        else:
            url = "https://demo1.testgold.dev/login"
        return url


    def _clear_page(self):
        self.page_state = {
            'clickables': [0], 
            'selectables': [], 
            'enterables': ["", "", ""]
            }

    def _check_enterables_fields(self):
        enterables = self.page_state.get('enterables')
        if enterables[0] == "email@example.com" and enterables[1] == "admin123":
            return True
        else:
            return False



if __name__ == "__main__":

    driver = SeleniumWebDriver(init_url_address="https://demo1.testgold.dev/login")
    elements = driver.get_site_elements()
    print(elements)
    #driver.action_on_element()
    driver.action_on_element("ENTER", 0)
    driver.action_on_element("ENTER", 1)
    driver.action_on_element("CLICK", 0)
    
    while len(driver.get_site_elements()['clickables']) == 0:
        time.sleep(10)
    elements = driver.get_site_elements()
    print(elements)

    driver.action_on_element("CLICK", 0)
    