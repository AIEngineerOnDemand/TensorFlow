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


## Understanding TensorFlow Metrics

In TensorFlow, metrics are instances of the `tf.keras.metrics.Metric` class, which you can use to track the progress of your training and testing loops. Here's a brief explanation of the metrics used in this code:

### `tf.keras.metrics.Mean`

`tf.keras.metrics.Mean` computes the (weighted) mean of the given values. In this case, it's used to track the average loss during training and testing. The `name` argument is used to give the metric a name, which can be useful for logging.

Here's how it's used:

```python
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')
````

In this example, `train_loss` and `test_loss` are metrics that compute the mean training and testing loss, respectively.

### `tf.keras.metrics.SparseCategoricalAccuracy`

`tf.keras.metrics.SparseCategoricalAccuracy` calculates how often predictions match integer labels. It's used to track the accuracy of the model during training and testing. The `name` argument is used to give the metric a name, which can be useful for logging.

Here's how it's used:

```python
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
```

In this example, `train_accuracy` and `test_accuracy` are metrics that compute the mean training and testing accuracy, respectively.

In summary, `tf.keras.metrics.Mean` and `tf.keras.metrics.SparseCategoricalAccuracy` are used to track the progress of the training and testing loops.


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

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

 EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_state()
  train_accuracy.reset_state()
  test_loss.reset_state()
  test_accuracy.reset_state()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result():0.2f}, '
    f'Accuracy: {train_accuracy.result() * 100:0.2f}, '
    f'Test Loss: {test_loss.result():0.2f}, '
    f'Test Accuracy: {test_accuracy.result() * 100:0.2f}'
  )
         
``` 

This decorator tells TensorFlow to compile the function using TensorFlow's graph mode, which can provide significant speedups.



## Training Models with the `tf.estimator` API

The `tf.estimator` API is a high-level TensorFlow API that greatly simplifies machine learning programming. It encapsulates training, evaluation, prediction, and export for serving. While `tf.keras` models can be trained directly using their built-in `fit` method, they can also be converted to an `Estimator` object and trained using the `tf.estimator` API.

Here's how you can convert a `tf.keras.Model` to an `Estimator`:

```python
import tensorflow as tf
from tensorflow.keras import layers

# Define a Keras model
model = tf.keras.models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Convert the Keras model to an Estimator
estimator = tf.keras.estimator.model_to_estimator(model)
```

In this example, a `tf.keras.Model` is first defined and compiled. Then, it's converted to an `Estimator` using the `tf.keras.estimator.model_to_estimator` function.

estimator = tf.keras.estimator.model_to_estimator(keras_model)

## Understanding `model_to_estimator`

The `model_to_estimator` method is a utility function provided by TensorFlow that converts a `tf.keras.Model` to a `tf.estimator.Estimator`. This allows you to leverage the simplicity and flexibility of Keras for model definition and debugging, while benefiting from the distributed computing capabilities of `tf.estimator`.

Here's a basic usage example:

```python
keras_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1)
])
keras_model.compile(optimizer='adam', loss='mse')

estimator = tf.keras.estimator.model_to_estimator(keras_model)
```
In this example, a Sequencial Keras model is defined and compiled, and then converted to an `Estimator` using `model_to_estimator`.

Useful Estimator Types for Machine Learning Problems
While model_to_estimator allows you to convert any Keras model to an Estimator, TensorFlow also provides several pre-made Estimators for common machine learning tasks. These include:

- tf.estimator.LinearClassifier: Constructs a linear classification model.
- tf.estimator.DNNClassifier: Constructs a neural network classification model.
- tf.estimator.DNNLinearCombinedClassifier: Constructs a neural network and linear combined classification model.
- tf.estimator.LinearRegressor: Constructs a linear regression model.
- tf.estimator.DNNRegressor: Constructs a neural network regression model.
- tf.estimator.DNNLinearCombinedRegressor: Constructs a neural network and linear combined regression model.
These pre-made Estimators can simplify the process of creating machine learning models, especially for common tasks like classification and regression.
Once you have an `Estimator`, you can train it using the `train` method:

```python
# Define the input function
def input_fn():
    # Generate training data here...
    return dataset

# Train the Estimator
estimator.train(input_fn, steps=1000)
```

In this example, an input function is defined that returns a `tf.data.Dataset` object. This dataset is used to train the `Estimator`.
# Evaluate the Estimator
```python
eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10)

print('Eval result: {}'.format(eval_result))
```
In summary, `tf.keras.estimator.model_to_estimator` provides a way to convert a `tf.keras.Model` to an `Estimator`, which can then be trained using the `tf.estimator` API.

## Common Class Methods of the `tf.estimator.Estimator`

The `tf.estimator.Estimator` class provides several methods for training, evaluating, and making predictions with a model. Here are some of the most commonly used methods:

### `train`

The `train` method trains a model for a fixed number of steps.

```python
def input_fn():
    # Generate training data here...
    return dataset

estimator.train(input_fn, steps=1000)
```
In this example, the train method is used to train the Estimator for 1000 steps.

### `evaluate`


The evaluate method evaluates the model's performance.
```python
def input_fn():
    # Generate test data here...
    return dataset

eval_result = estimator.evaluate(input_fn=input_fn, steps=10)
print('Eval result: {}'.format(eval_result))
```
In this example, the evaluate method is used to evaluate the Estimator on some test data.

### `predict`
The predict method makes predictions for a batch of instances.
```python
def input_fn():
    # Generate prediction data here...
    return dataset

predictions = list(estimator.predict(input_fn))
```
In this example, the predict method is used to make predictions for a batch of instances.

### `export_saved_model`
The export_saved_model method exports the model to the SavedModel format.
```python
estimator.export_saved_model('export', serving_input_receiver_fn)
```
In this example, the export_saved_model method is used to export the Estimator to the SavedModel format, which can be used for serving.

In summary, the tf.estimator.Estimator class provides several methods for training, evaluating, making predictions with, and exporting a model. 



## Defining a Hypermodel for Hyperparameter Tuning

When setting up a model for hyperparameter tuning, you not only define the model architecture, but also the hyperparameter search space. The model you set up for hyperparameter tuning is called a hypermodel.

There are two main ways to define a hypermodel:

1. Using a model builder function.
2. Subclassing the `HyperModel` class of the Keras Tuner API.

In addition, the Keras Tuner provides two pre-defined `HyperModel` classes - `HyperXception` and `HyperResNet` for computer vision applications.

In the provided code, a model builder function is used to define an image classification model. The model builder function returns a compiled model and uses hyperparameters defined inline to hypertune the model.

Here's a breakdown of the `model_builder` function:

```python
def model_builder(hp):
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(10))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

  ````

  In this function:

A Sequential model is created with a Flatten layer as the input layer, which flattens the input data to a 1D array.
The number of units in the first Dense layer is a hyperparameter, which is set to an integer value between 32 and 512.
The learning rate for the Adam optimizer is also a hyperparameter, which is set to one of three possible values: 0.01, 0.001, or 0.0001.
The model is compiled with the Adam optimizer, the SparseCategoricalCrossentropy loss, and accuracy as the metric.
The compiled model is returned.

