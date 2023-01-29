from sonic.review import plot_result
from sonic.gpu import setup_gpus
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
setup_gpus()


def train(options={}, dataset=(), model_builder=None):

    if not model_builder:
        print("no model builder provided")
        return

    ds_train = dataset[0]
    ds_val = dataset[1]
    class_names = dataset[2]

    model, model_name = model_builder(
        height=options["height"],
        width=options["width"],
        channels=options["channels"],
        num_classes=len(class_names),
    )

    if options["show_summary"]:
        model.summary()

    callbacks = []

    # if we want to save the model, add the callback
    if options["save_model"]:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f".models/{model_name}.keras",
                save_best_only=True,
                monitor="val_loss",
            )
        )

    # if we want to do some early stopping, add the callback
    if options["early_stop"]:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=2))

    history = model.fit(
        ds_train[0],
        ds_train[1],
        validation_data=(ds_val[0], ds_val[1]),
        epochs=options["epochs"],
        callbacks=callbacks,
    )

    if options["plot_results"]:
        plot_result(history)
