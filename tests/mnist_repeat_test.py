import pytest

# import time
# import threading
import sys
import os
sys.path.append(os.getcwd() + '/MNIST')
from VisualizeMNIST_repeat import Scribble


class TestScribble:

    @pytest.fixture()
    def scribble(request):
        return Scribble(modelName=os.getcwd() + "/MNIST/20160818_MNIST.model")

    def test_repeatJudge(self, scribble):
        # assert 5 == scribble.val.index(max(scribble.val))
        return True
