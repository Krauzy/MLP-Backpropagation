# MultiLayer Perceptron - Backpropagation
 A Multilayer Perceptron Neural Network using Backpropagation Architecture
 
### Perceptron
 [Perceptron](https://en.wikipedia.org/wiki/Perceptron) is a type of artificial neural network invented in 1958 by [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt) at the [Cornell Aeronautical Laboratory](https://en.wikipedia.org/wiki/Calspan). It can be seen as the simplest type of feedforward neural network: a linear classifier
 
 <img src="https://miro.medium.com/max/468/1*GSnd8cg2lKix33zkr3AUxA.png" width="40%">
 
 ### Multilayer Perceptron
 A [multilayer perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron) is a class of feedforward artificial neural network (ANN),  sometimes strictly used to refer to networks composed of multiple layers of perceptrons (with threshold activation)
 
 <img src="https://miro.medium.com/max/600/1*xxZXeKfVKTRqh54t10815A.jpeg" width="40%">
 
 ### Backpropagation
 In machine learning, [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) is a widely used algorithm for training feedforward neural networks. Generalizations of backpropagation exist for other artificial neural networks (ANNs), and for functions generally. These classes of algorithms are all referred to generically as "backpropagation". In fitting a neural network, backpropagation computes the [gradient](https://en.wikipedia.org/wiki/Gradient) of the loss function with respect to the weights of the network for a single input–output example, and does so efficiently, unlike a naive direct computation of the gradient with respect to each weight individually.
 
 <img src="https://i.stack.imgur.com/7Ui1C.png" width="60%">
 
 ## Activation Functions

 | Function | Activation | Gradient |
 | ----- | ----- | ----- |
 | Linear | `1 / net` | `1 / 10` |
 | Logistics | `1 / (1 + (e ^ -net)` | `(e ^ -net) * (1 - e ^ -net)` |
 | Hiperbolic Tangent | `tanh(net)` | `1 - tanh(net) ^ 2` |
 
 ## Dependencies
 
 Pipfile
 
 ```sh
 [packages]
 pandas = "*"
 streamlit = "*"
 numpy = "*"

 [dev-packages]

 [requires]
 python_version = "3.8"
 ```
 
 ## Init
 
 1. Install `streamlit` package;
 2. Open shell terminal (anaconda, pycharm, etc.);
 3. Run `streamlit run main.py` in root of project;

 
 ## Licences
 
 [MIT](https://github.com/Krauzy/MLP-Backpropagation/blob/main/LICENSE)
