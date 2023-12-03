# Load modules
print("Loading modules...")

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# %% [markdown]
# ## Prepare the data

# %%
# num_classes = 100
# input_shape = (32, 32, 3)

# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

# print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
# print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")


# %% [markdown]
# ## Configure the hyperparameters

# %%
input_shape = (512, 512, 1)

print("Reading in data...")
# Read in data
x_train_singles = np.load("/mnt/home/taftkyle/indiv_data/singles/raw_single_train.npy")
# y_train_singles = np.load("/mnt/home/taftkyle/indiv_data/singles/single_train_label.npy")
# x_test_singles = np.load("/mnt/home/taftkyle/indiv_data/singles/raw_single_test.npy")[0:100]
# y_test_singles = np.load("/mnt/home/taftkyle/indiv_data/singles/single_test_label.npy")[0:100]

# new_y_train_singles = []
# for i in range(len(y_train_singles)):
#     cord = np.unravel_index(np.argmax(y_train_singles[i], axis=None), y_train_singles[i].shape)[0:2]
#     cord = np.array([20*cord[1], 20*(511-cord[0])]) # convert to energy
#     new_y_train_singles.append(np.pad(cord, (0, 2), 'constant', constant_values=(-1,-1))) # add padding of -1 energies (unphysical)

# new_y_test_singles = []
# for i in range(len(y_test_singles)):
#     cord = np.unravel_index(np.argmax(y_test_singles[i], axis=None), y_test_singles[i].shape)[0:2]
#     cord = np.array([20*cord[1], 20*(511-cord[0])]) # convert to energy
#     new_y_test_singles.append(np.pad(cord, (0, 2), 'constant', constant_values=(-1,-1))) # add padding of -1 energies (unphysical)

# y_train_singles = np.array(new_y_train_singles)
# y_test_singles = np.array(new_y_test_singles)

x_train_double = np.load("/mnt/home/taftkyle/indiv_data/doubles/raw_double_train.npy")
# y_train_double = np.load("/mnt/home/taftkyle/indiv_data/doubles/double_train_label.npy")
# x_test_double = np.load("/mnt/home/taftkyle/indiv_data/doubles/raw_double_test.npy")[0:100]
# y_test_double = np.load("/mnt/home/taftkyle/indiv_data/doubles/double_test_label.npy")[0:100]

# new_y_train_double = []
# for i in range(len(y_train_double)):
#     cord = np.unravel_index(np.argmax(y_train_double[i], axis=None), y_train_double[i].shape)[0:2]
#     # find the second max
#     y_train_double[i][cord[0], cord[1]] = 0
#     cord2 = np.unravel_index(np.argmax(y_train_double[i], axis=None), y_train_double[i].shape)[0:2]
#     new_y_train_double.append([20*cord[1], 20*(511-cord[0]), 20*cord2[1], 20*(511-cord2[0])])

# new_y_test_double = []
# for i in range(len(y_test_double)):
#     cord = np.unravel_index(np.argmax(y_test_double[i], axis=None), y_test_double[i].shape)[0:2]
#     # find the second max
#     y_test_double[i][cord[0], cord[1]] = 0
#     cord2 = np.unravel_index(np.argmax(y_test_double[i], axis=None), y_test_double[i].shape)[0:2]
#     new_y_test_double.append([20*cord[1], 20*(511-cord[0]), 20*cord2[1], 20*(511-cord2[0])])

# y_train_double = np.array(new_y_train_double)
# y_test_double = np.array(new_y_test_double)

# %%
#Combine the data
x_train = np.concatenate((x_train_singles, x_train_double))
y_train = np.load("y_train.npy")
# y_train = np.concatenate((y_train_singles, y_train_double))
# x_test = np.concatenate((x_test_singles, x_test_double))
# y_test = np.concatenate((y_test_singles, y_test_double))

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
# print("x_test shape:", x_test.shape)
# print("y_test shape:", y_test.shape)

# np.save("y_train.npy", y_train)

#Shuffle data
indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]




# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='linear'))  # (x,y) coordinates

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.MeanSquaredError(),
    metrics=[
        keras.metrics.MeanSquaredError(name="mse"),
        keras.metrics.MeanAbsoluteError(name="mae"),
    ],
)




#Train model

print("Training model...")


checkpoint_filepath = "bestCNNmodel.h5"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_filepath,
    monitor="val_mae",
    mode = 'min',
    save_best_only=True,
)

history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=32,
        epochs=100,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
)


print("Model trained.")

#Save model
model.save('cnn10')

#Save loss curves
plt.figure()
plt.plot(history.history['loss'], label = "training loss")
plt.plot(history.history['val_loss'], label = "validation loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.savefig('trainlossCNN100.png')





