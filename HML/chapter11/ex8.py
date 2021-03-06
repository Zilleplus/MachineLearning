import tensorflow.keras as keras
import os

simple_model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32, 32, 3]),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(10, activation="softmax")
])

model_BN = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32, 32, 3]),
    keras.layers.BatchNormalization(),  # input normalization
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dense(10, activation="softmax")
])


model_BN_selu = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32, 32, 3]),
    keras.layers.BatchNormalization(),  # input normalization
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.Dense(10, activation="softmax")
])

drop_rate = 0.2
model_BN_selu_reg = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[32, 32, 3]),
    keras.layers.BatchNormalization(),  # input normalization
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.Dense(10, activation="softmax")
])


root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
earlystopping_cb = keras.callbacks.EarlyStopping(patience=5)  # restore_best_weights=True,
callbacks = [earlystopping_cb]

model = model_BN_selu_reg

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

learning_rate = 1e-5
# optimizer = keras.optimizers.SGD(lr=learning_rate)
optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])
model.fit(
    x_train,
    y_train,
    epochs=100,
    validation_split=0.2,
    verbose=1,
    callbacks=callbacks
  )
