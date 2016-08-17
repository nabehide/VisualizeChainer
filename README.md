
# TkinterChainer
chainerで学習させたニューラルネットワークを使ってみて学習できたか確認する

## MNIST
マウス入力できるcanvasはTkinterで作成されています。
canvasのピクセルデータを学習済みのニューラルネットワークに入力して数字を認識します。

    1. Tkinterのcanvasに数字を書く
    2. judgeボタンを押すと数字を認識して結果を右側に表示される

学習済みモデルは下記です。
MNISTディレクトリ下に置いてください。
https://github.com/nabehide/VisualizeChainer/releases/download/v1.0.0/20160818_MNIST.model.zip

![](https://cloud.githubusercontent.com/assets/18606082/17755347/87c44a9e-6514-11e6-826d-1ebf02304d21.png)

