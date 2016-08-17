# -*- coding:utf-8 -*-

import Tkinter as tkinter # Python 2
#import tkinter # Python 3

from PIL import Image, ImageDraw, ImageOps
import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers

# name
fileName = "outfile.png" # save png name
modelName = '20160818_MNIST.model'

# size
window_width = 300
window_height = 300
canvas_width = 28*10
canvas_height = 28*10
button_width = 5
button_height = 1
BAR_SPACE = 3
BAR_WIDTH = 30

# color depth
draw_depth = int(50. /100.*255) #50%

class MLP(chainer.Chain):

    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

class Scribble():

    def on_pressed(self, event):
        self.sx = event.x
        self.sy = event.y

    def on_dragged(self, event):
        # draw surface canvas
        self.canvas.create_line(self.sx, self.sy, event.x, event.y,
                                width = 5,
                                tag = "draw")

        # draw hidden canvas
        ## I was going to write the code which could make gradation like MNIST.
        ## However, it seems to work without below codes for the moment.
        """
        draw_width = 1
        for i in range(draw_width):
            color = int(255/(draw_width+1))
            self.draw.line(((self.sx+1,self.sy+1),(event.x+1,event.y+1)),(color,color,color),width/28)
            self.draw.line(((self.sx-1,self.sy-1),(event.x-1,event.y-1)),(color,color,color),width/28)
        """
        self.draw.line(((self.sx,self.sy),(event.x,event.y)),(draw_depth,draw_depth,draw_depth),window_width/28*3)

        # store the position in the buffer
        self.sx = event.x
        self.sy = event.y

    def judge(self):
        # save png and convert to values
        self.image1.save(fileName)
        input_image = Image.open(fileName)
        gray_image = ImageOps.grayscale(input_image)
        pr_resize = np.array(gray_image.resize((28,28)).getdata()).astype(np.float32)
        pr_resize /= 255.
        pr_resize = 1. - pr_resize # invert black and white

        # regit recognition using the neural network(NN)
        h1 = F.dropout(F.relu(self.mlp.l1(chainer.Variable(pr_resize.reshape(1,784)))),train=False)
        h2 = F.dropout(F.relu(self.mlp.l2(h1)),train=False)
        y = self.mlp.l3(h2)

        # show the result
        self.result.delete("result") # delete the previous data
        for i in range(10):
            # format the value 
            val = max(np.array(y.data)[0][i],0.) / np.max(np.array(y.data))

            # show the bar
            self.result.create_rectangle(30,i*BAR_WIDTH+BAR_SPACE,30+int((window_width-60)*val),(i+1)*BAR_WIDTH,tag="result")

            # show the number and the NN's output
            self.result.create_text(15,i*BAR_WIDTH+BAR_SPACE+BAR_WIDTH/2,text=str(i),tag="result")
            self.result.create_text(window_width-15,i*BAR_WIDTH+BAR_SPACE+BAR_WIDTH/2,text=str("%.2f"%val),tag="result")

    def clear(self):
        # clear the surface canvas
        self.canvas.delete("draw")

        # clear(initialize) the hidden canvas
        self.image1 = Image.new("RGB",(window_width,window_height),(255,255,255))
        self.draw = ImageDraw.Draw(self.image1)

        # clear the result
        self.result.delete("result")

    def create_window(self):
        window = tkinter.Tk()

        # canvas frame
        canvas_frame = tkinter.LabelFrame(window, bg = "white",
                                     text = "canvas",
                                     width = window_width, height = window_height,
                                     relief = 'groove', borderwidth = 4)
        canvas_frame.pack(side = tkinter.LEFT)
        self.canvas = tkinter.Canvas(canvas_frame, bg = "white",
                                     width = canvas_width, height = canvas_height,
                                     relief = 'groove', borderwidth = 4)
        self.canvas.pack()
        quit_button = tkinter.Button(canvas_frame, text = "exit",
                                     command = window.quit)
        quit_button.pack(side = tkinter.RIGHT)
        judge_button = tkinter.Button(canvas_frame, text = "judge",
                                      width = button_width, height = button_height,
                                      command = self.judge)
        judge_button.pack(side = tkinter.LEFT)
        clear_button = tkinter.Button(canvas_frame, text = "clear",
                                      command = self.clear)
        clear_button.pack(side = tkinter.LEFT)
        self.canvas.bind("<ButtonPress-1>", self.on_pressed)
        self.canvas.bind("<B1-Motion>", self.on_dragged)

        # result frame
        result_frame = tkinter.LabelFrame(window, bg = "white",
                                     text = "result",
                                     width = window_width, height = window_height,
                                     relief = 'groove', borderwidth = 4)
        result_frame.pack(side = tkinter.RIGHT)
        self.result = tkinter.Canvas(result_frame, bg = "white",
                                     width = window_width,
                                     height = window_height)
                                     #height = (BAR_WIDTH+BAR_SPACE)*10)
        self.result.pack()

        return window

    def __init__(self):
        self.window = self.create_window()

        # set canvas
        self.image1 = Image.new("RGB",(window_width,window_height),(255,255,255))
        self.draw = ImageDraw.Draw(self.image1)

        # set neural network model
        self.mlp = MLP(784,1000,10)
        model = L.Classifier(self.mlp)
        serializers.load_hdf5(modelName, model)
            
    def run(self):
        self.window.mainloop()

def main():
    Scribble().run()
		
if __name__ == '__main__':
	main()

