from keras import layers
import tensorflow as tf
import keras as keras
import os
import shutil
import pandas as pd
import tarfile
import numpy as np

ARCHIVE_NAME = "cora"
hidden_units = [32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256


def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []
    for unit in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(rate=dropout_rate))
        fnn_layers.append(layers.Dense(units=unit, activation=tf.nn.gelu))
    return keras.Sequential(fnn_layers, name=name)


def create_baseline_model(hidden_units, num_classes, num_features, dropout_rate=0.2):
    inputs = layers.Input(shape=num_features, name="input_layer")
    x = create_ffn(hidden_units, dropout_rate, name="ffn_block1")(inputs)
    for i in range(4):
        x1 = create_ffn(hidden_units, dropout_rate, name=f"ffn_block{i + 2}")(x)
        x = layers.Add(name=f"fskip_connection{i + 2}")([x, x1])
    logits = layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs=inputs, outputs=logits, name="baseline_model")


def run_experiment(model, x_train, y_train):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )

    return model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.15)


tar_file = tarfile.open(ARCHIVE_NAME + ".tgz", "r")
try:
    tar_file.extractall()
    citations = pd.read_csv(
        os.path.join(ARCHIVE_NAME, "cora.cites"),
        sep="\t",
        header=None,
        names=["target", "source"],
    )
    column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
    papers = pd.read_csv(
        os.path.join(ARCHIVE_NAME, "cora.content"), sep="\t", header=None, names=column_names,
    )
finally:
    if os.path.exists(ARCHIVE_NAME):
        shutil.rmtree(ARCHIVE_NAME)

# converting categorical val to integer

class_idx = {name: id for id, name in enumerate(sorted(papers.subject.unique()))}
paper_idx = {name: id for id, name in enumerate(sorted(papers.paper_id.unique()))}

papers.paper_id = papers.paper_id.apply(lambda name: paper_idx[name])
citations.source = citations.source.apply(lambda name: paper_idx[name])
citations.target = citations.target.apply(lambda name: paper_idx[name])
papers.subject = papers.subject.apply(lambda name: class_idx[name])

# splitting data to train/test

train_data, test_data = [], []

for _, group_data in papers.groupby("subject"):
    random_selection = np.random.rand(len(group_data.index)) <= 0.5
    train_data.append(group_data[random_selection])
    test_data.append(group_data[~random_selection])

train_data = pd.concat(train_data).sample(frac=1)
test_data = pd.concat(test_data).sample(frac=1)

feature_names = set(papers.columns) - {"paper_id", "subject"}
num_features = len(feature_names)
num_classes = len(class_idx)

x_train = train_data[feature_names].to_numpy()
x_test = test_data[feature_names].to_numpy()

y_train = train_data.subject
y_test = test_data.subject

baseline_model = create_baseline_model(hidden_units, num_classes, num_features, dropout_rate)
run_experiment(baseline_model, x_train, y_train)

_, test_accuracy = baseline_model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")