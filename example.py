# x11vnc -forever -display :0 -noxdamage -repeat -rfbport 5900 -shared
import gym, gym_mupen64plus
import numpy as np
import tensorflow as tf
import random
from skimage.transform import resize
from collections import deque
from gym import wrappers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

env = gym.make('Mario-Kart-Luigi-Raceway-v0')

# Constants defining our neural network
INPUT_WIDTH = 200
INPUT_HEIGHT = 66
INPUT_CHANNELS = 3
OUT_SHAPE = 5

gamma = 0.8
REPLAY_MEMORY = 50000

def resize_image(img):
    im = resize(img, (66, 200, 3))
    im_arr = im.reshape((66, 200, 3))
    return im_arr

def prepare_image(im_arr):
    im_arr = im_arr.reshape((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
    im_arr = np.expand_dims(im_arr, axis=0)
    return im_arr

def customized_loss(y_true, y_pred, loss='euclidean'):
    # Simply a mean squared error that penalizes large joystick summed values
    if loss == 'L2':
        L2_norm_cost = 0.001
        val = K.mean(K.square((y_pred - y_true)), axis=-1) \
            + K.sum(K.square(y_pred), axis=-1) / 2 * L2_norm_cost
    # euclidean distance loss
    elif loss == 'euclidean':
        val = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    return val


def create_model(keep_prob=0.6):
    model = Sequential()

    # NVIDIA's model
    model.add(BatchNormalization(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)))
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    drop_out = 1 - keep_prob
    model.add(Dropout(drop_out))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(OUT_SHAPE, activation='softsign', name="predictions"))

    return model


def dqn_replay_train(mainDQN, targetDQN, train_batch):
    x_train = []
    y_train = []

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        x_train.append(state)
        # terminal?
        if done:
            y_train.append(reward)
        else:
            # Q[0, action] = reward + gamma * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]
            y_train.append(reward + gamma * np.max(targetDQN.predict(prepare_image(next_state)))) ##### use normal one for now

    # Train our network using target and predicted Q values on each episode
    x = np.asarray(x_train)
    y = np.asarray(y_train).reshape((len(y_train), 1))
    return mainDQN.fit(x, y)

def get_angle(steer):
    if steer == 0:
        return -80
    if steer == 1:
        return -40
    if steer == 2:
        return 0
    if steer == 3:
        return 40
    if steer == 4:
        return 80

def main():
    max_episodes = 5000
    # store the previous observations in replay memory
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = create_model()
        mainDQN.compile(loss=customized_loss, optimizer=optimizers.Adam(lr=0.0001))

        targetDQN = create_model()
        targetDQN.compile(loss=customized_loss, optimizer=optimizers.Adam(lr=0.0001))

        for episode in range(max_episodes):
            e = 0.5
            done = False
            step_count = 0
            state = env.reset()
            state = resize_image(state)
            while not done:
                if np.random.rand(1) < e:
                    steer = random.randint(0, 4)
                else:
                    # Choose an action by greedily from the Q-network
                    actions = mainDQN.predict(prepare_image(state), batch_size=1)[0]
                    print actions
                    steer = np.argmax(actions)

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step([get_angle(steer), 0, 1, 0, 0])
                next_state = resize_image(next_state)

                # Save the experience to our buffer
                replay_buffer.append((state, steer, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                      replay_buffer.popleft()

                state = next_state
                step_count += 1

                if len(replay_buffer) > 100:
                    minibatch = random.sample(replay_buffer, 10)
                    dqn_replay_train(mainDQN, targetDQN, minibatch)

                if step_count % 1000 == 999:
                    targetDQN.set_weights(mainDQN.get_weights())

if __name__ == "__main__":
    main()