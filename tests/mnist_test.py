import pytest

import sys
import os
sys.path.append(os.getcwd() + '/MNIST')
from VisualizeMNIST import Scribble


# class TestScribble(TestCase):
class TestClass:

    @pytest.fixture()
    def scribble(request):
        return Scribble(modelName=os.getcwd() + "/MNIST/20160818_MNIST.model")

    def test_on_pressed(self, scribble):
        return True

    def test_on_dragged(self):
        return True

    def test_judge(self, scribble):
        scribble.judge()
        assert 5 == scribble.val.index(max(scribble.val))

    def test_clear(self, scribble):
        scribble.clear()
        return True

    def test_create_window(self, scribble):
        scribble.create_window()
        return True

    def test_run(self, scribble):
        return True
