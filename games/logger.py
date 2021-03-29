import os
import sys


class Logger():
    def __init__(self, path="_logger.log"):
        self.fp = open(path, "wt")

    def log(self, msg=""):
        self.fp.write("{}\n".format(msg))
        self.fp.flush()
        if len(msg) > 0:
            print("log: {}".format(msg))
        else:
            print()

