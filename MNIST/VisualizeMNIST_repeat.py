# -*- coding:utf-8 -*-

import time
import threading

from VisualizeMNIST import Scribble as sc


class Scribble(sc):

    def repeatJudge(self):
        """
        Sub process.
        Repeat to recognize digit.
        """
        while not self.stop_event.is_set():
            try:
                self.judge()
            except RuntimeError:
                return
            except ValueError:
                print('Please check model file')
                self.window.quit()
                return
            time.sleep(0.5)

    def __init__(self, modelName='20160818_MNIST.model'):
        """
        Initial procedure.
        Create GUI, set model file and processes start.
        """
        super(Scribble, self).__init__(modelName)

        # sub process starts
        self.stop_event = threading.Event()
        th_me = threading.Thread(target=self.repeatJudge)
        th_me.start()

    def __exit__(self):
        """
        Sub process stops when GUI closes.
        """
        self.stop_event.set()


def main():
    Scribble().run()


if __name__ == '__main__':
    main()
