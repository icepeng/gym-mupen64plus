# x11vnc -forever -display :0 -noxdamage -repeat -rfbport 5900 -shared
import gym, gym_mupen64plus
import numpy as np
import tensorflow as tf
import random
from skimage.transform import resize
from collections import deque
from gym import wrappers

env = gym.make('Mario-Kart-Discrete-Luigi-Raceway-v0')

# Constants defining our neural network
input_size = 66*200*3        #####change input_size - 224*256*3 acquired from ppaquette_gym_super_mario/nes_env.py
output_size = env.action_space.n                                                                     #####meaning of output can be found at ppaquette_gym_super_mario/wrappers/action_space.py

dis = 0.9
REPLAY_MEMORY = 5000

def resize_image(img):
    im = resize(img, (66, 200, 3))
    im_arr = im.reshape((66, 200, 3))
    return im_arr

class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=10, l_rate=1e-1):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")

            # First layer of weights
            W1 = tf.get_variable("W1", shape=[self.input_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.tanh(tf.matmul(self._X, W1))

            # Second layer of Weights
            W2 = tf.get_variable("W2", shape=[h_size, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer())

            # Q prediction
            self._Qpred = tf.matmul(layer1, W2)

        # We need to define the parts of the network needed for learning a policy
        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        # Loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        vec = resize_image(state)
        x = np.reshape(vec, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})

def ddqn_replay_train(mainDQN, targetDQN, train_batch):
    '''
    Double DQN implementation
    :param mainDQN: main DQN
    :param targetDQN: target DQN
    :param train_batch: minibatch for train
    :return: loss
    '''
    x_stack = np.empty(0).reshape(0, mainDQN.input_size)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        if state is None:                                                               #####why does this happen?
            print("None State, ", action, " , ", reward, " , ", next_state, " , ", done)
        else:
            Q = mainDQN.predict(state)

            # terminal?
            if done:
                Q[0, action] = reward
            else:
                # Double DQN: y = r + gamma * targetDQN(s')[a] where
                # a = argmax(mainDQN(s'))
                Q[0, action] = reward + dis * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]
                # Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state)) #####use normal one for now

            y_stack = np.vstack([y_stack, Q])
            x_stack = np.vstack([x_stack, resize(state, (66, 200, 3)).reshape(-1, mainDQN.input_size)])   #####change shape to fit to super mario

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)

def get_copy_var_ops(dest_scope_name="target", src_scope_name="main"):

    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def bot_play(mainDQN, env=env):
    # See our trained network in action
    state = env.reset()
    reward_sum = 0
    while True:
        env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break

def main():
    max_episodes = 5000
    # store the previous observations in replay memory
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = DQN(sess, input_size, output_size, name="main")
        targetDQN = DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()

        #initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(max_episodes):
            e = 1. / (((episode + 8) / 10) + 1)
            print episode, e
            done = False
            step_count = 0
            total_reward = 0
            state = env.reset()
            while not done:
                if np.random.rand(1) < e
                    action = env.action_space.sample()
                else:
                    # Choose an action by greedily from the Q-network
                    predictions = mainDQN.predict(state)
                    action = np.argmax(predictions)
                    # action = mainDQN.predict(state).flatten().tolist()                  #####flatten it and change it as a list
                    # for i in range(len(action)):                                        #####the action list has to have only integer 1 or 0
                    #     if action[i] > 0.5 :
                    #         action[i] = 1                                               #####integer 1 only, no 1.0
                    #     else:
                    #         action[i] = 0                                               #####integer 0 only, no 0.0

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                      replay_buffer.popleft()

                state = next_state
                step_count += 1
                total_reward += reward
                if step_count > 500:   # Good enough. Let's move on
                    break
            
            print total_reward
            if episode % 10 == 1: # train every 10 episode
                # Get a random batch of experiences
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = ddqn_replay_train(mainDQN, targetDQN, minibatch)

                print("Loss: ", loss)
                # copy q_net -> target_net
                sess.run(copy_ops)

        # See our trained bot in action
        env2 = wrappers.Monitor(env, 'gym-results', force=True)

        for i in range(200):
            bot_play(mainDQN, env=env2)

        env2.close()
        # gym.upload("gym-results", api_key="sk_VT2wPcSSOylnlPORltmQ")

if __name__ == "__main__":
    main()