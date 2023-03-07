import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

vocab_size = 1000
trunc_type='post'
padding_type='post'
loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
# optimizer_object = tf.keras.optimizers.Adam(learning_rate=0.1)    
optimizer_object='sgd'

def prepare_sequences(tokenizer,training_sentences,testing_sentences,training_labels,testing_labels):
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    max_length = max([len(i) for i in training_sequences])
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    training_labels_seq = tokenizer.texts_to_sequences(training_labels)
    max_length_lab = max([len(i) for i in training_labels_seq])
    testing_labels_seq = tokenizer.texts_to_sequences(testing_labels)

    training_lab_padded = pad_sequences(training_labels_seq, maxlen=max_length_lab, padding=padding_type, truncating=trunc_type)
    testing_lab_padded = pad_sequences(testing_labels_seq, maxlen=max_length_lab, padding=padding_type, truncating=trunc_type)

    training_padded = np.array(training_padded)
    training_labels = np.array(training_lab_padded)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_lab_padded)
    
    return training_padded,training_labels,testing_padded,testing_labels,max_length,max_length_lab


def model_init(vocab_size,embedding_dim,max_length,max_length_lab):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(max_length_lab)
        # tf.keras.layers.Dense(24, activation='sigmoid')
    ])

    model.compile(
        loss=loss_object,
        optimizer=optimizer_object,
        metrics=["accuracy"],
    )
    return model

def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)

    return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def model_compiler(model,training_padded,training_lab_padded,num_epochs,training_size,batch_size):
    train_loss_results = []
    train_accuracy_results = []
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.Accuracy()

    for epoch in range(num_epochs):
        
        # Training loop - using batches of 32
        for batch in range(int(training_size/batch_size)):
            # Optimize the model
            x= training_padded[(batch*batch_size):((batch+1)*batch_size)]
            y=training_lab_padded[(batch*batch_size):((batch+1)*batch_size)]
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            epoch_accuracy.update_state(y, model(x, training=True),sample_weight=None)

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        # if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))
        return train_loss_results,train_accuracy_results

def save_model(model):
    tf.keras.models.save_model(
        model,
        "NL_save",
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
    
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
  
def plot_graph2(train_loss_results,train_accuracy_results):
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    plt.show()