# Reference code for Tensorflow  



## Understanding Different Ways to Define Models in TensorFlow

In TensorFlow, there are two common ways to define models: using the `Sequential` API, and using the Model subclassing. Here's a brief explanation of the differences:

### `Sequential` API

The `Sequential` API is a way of creating deep learning models where each layer has exactly one input tensor and one output tensor. It's a simple stack of layers that can't represent arbitrary models.

`Sequential` models are created using the `Keras` API, which is included in TensorFlow. Here's an example:

```python
from tensorflow.keras import Sequential, layers

model = Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
```

In this example, the model is a simple stack of three dense layers. It's easy to understand and use, but it's not suitable for models with shared layers, multiple inputs, or multiple outputs.

## Model Subclassing with TensorFlow's Keras Model Class
Model subclassing is a way of creating models that gives more flexibility, at the cost of greater complexity. It involves defining a new class that inherits from the Model class, and overriding the __init__ and call methods.
The `Model` class in `tensorflow.keras` is a high-level API that is used for defining a neural network model. It groups layers into an object with training and inference features.
Here's an example:

```python
from tensorflow.keras import Model, layers

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.dense3 = layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

model = MyModel()
````

Here's another example with convolution nueral networks:

```python
from tensorflow.keras import Model, layers

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

# Create an instance of the model
model = MyModel()
```
## Difference Between `model.fit()` and `tf.GradientTape()` for Training

In TensorFlow, there are multiple ways to train a model. Two common methods are using the `fit` method of a compiled model, and using a `tf.GradientTape` to manually compute gradients. Here's a brief explanation of the differences:

## `model.fit()`

`model.fit()` is a high-level method provided by Keras (which is included in TensorFlow). It abstracts away many of the details of training a model, making it very easy to use. 

To use `model.fit()`  is a high-level method provided by Keras (which is included in TensorFlow). It abstracts away many of the details of training a model, making it very easy to use. You first need to compile your model with a specified optimizer, loss function, and metrics. Then, you simply call `model.fit()` with your training data and labels, and Keras handles the rest.

Here's an example:

```python
model = K.Sequential([
    data_normalizer,
    Dense(64,  activation='relu'),
    Dense(32,  activation='relu'),
    Dense(1,  activation=None)
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
```

In this example, the model is trained for 10 epochs using the Adam optimizer and mean squared error loss. The accuracy of the model is also tracked.

## `tf.GradientTape()`
`tf.GradientTape()` is a lower-level method that provides more control over the training process. It allows you to manually compute the gradients of the loss with respect to the model's parameters, which you can then use to update the model's weights.
`tf.GradientTape()` is a context manager provided by TensorFlow for automatic differentiation - the process of computing gradients of a computation with respect to some inputs, usually `tf.Variable`s. 

Automatic differentiation is a key part of many machine learning algorithms, as it allows us to optimize our models with respect to some loss function. 

When operations are performed within this context, TensorFlow "records" them onto a "tape". Then, TensorFlow uses that tape and the gradients associated with each recorded operation to compute the gradients of a recorded computation using reverse mode differentiation.
 
`tf.GradientTape()` provides a way to compute gradients for custom training loops, giving you more control over your machine learning model training process

Here's an example:

```python
with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)

gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

In this example, the model's predictions are computed within the context of a tf.GradientTape block. This allows TensorFlow to "record" the operations that are used to compute the loss, so it can then compute the gradients of the loss with respect to the model's parameters. These gradients are then used to update the model's weights.

In summary, model.fit() is a high-level, easy-to-use method for training a model, while tf.GradientTape() provides more control and is useful for more complex training loops. 





## Explanation of TensorFlow Training Step

The `train_step` function is a single step in the training of a TensorFlow model. Here's what each part does:


```python
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
``` 

This decorator tells TensorFlow to compile the function using TensorFlow's graph mode, which can provide significant speedups.





