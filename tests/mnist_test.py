import pytest
from unittest import TestCase

import sys
import os
sys.path.append(os.getcwd() + '/MNIST')
from VisualizeMNIST import Scribble


class TestScribble(TestCase):

    @pytest.mark.skipif("True")
    def test_on_pressed(self):
        self.assertTrue(True)

    @pytest.mark.skipif("True")
    def test_on_dragged(self):
        self.assertTrue(True)

    def test_judge(self):
        scribble = Scribble(
            modelName=os.getcwd() + "/MNIST/20160818_MNIST.model"
        )
        scribble.judge()
        assert 5 == scribble.val.index(max(scribble.val))


if __name__ == '__main__':
    pytest.main()
