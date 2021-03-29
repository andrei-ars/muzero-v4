import sys
import random
import string


class DataGenerator():
    
    def __init__(self, driver):
        self.driver = driver

    def infer(self, element, html_code=None):
        """
        element : PageElement, has attr params like
         {'id': 1, 'name': 'password', 'type': 'password', 
         'text': 'Your password...', 'selenium_type': 'input', 'xpath': '/html/body/div/main/div/div/section/form/div[1]/div[2]/input'}
        html_code : str
        """
        #print(element.params)
        if type(element) == str:
            element_name = element
        else:
            element_name = element.name

        maxlength = 10
        length = random.randint(1, maxlength)
        letters = string.ascii_lowercase + string.digits + string.punctuation
        random_str = ''.join(random.choice(letters) for i in range(length))
        
        if element_name == "email":
            result = "email@example.com"
        elif element_name == "password":
            result = "admin123"
        else:
            result = ""
        return result


if __name__ == "__main__":

    datagen =  DataGenerator()
    data = datagen.infer(None, None)
    print("data:", data)