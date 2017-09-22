import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque

EPISDOE = 10000
STEP = 10000
ENV_NAME = 'MountainCar-v0'
BATCH_SIZE = 32
INIT_EPSILON = 1.0
FINAL_EPSILON = 0.1
REPLAY_SIZE = 50000
TRAIN_START_SIZE = 200
GAMMA = 0.9
def get_weights(shape):
    weights = tf.truncated_normal( shape = shape, stddev= 0.01 )
    return tf.Variable(weights)

def get_bias(shape):
    bias = tf.constant( 0.01, shape = shape )
    return tf.Variable(bias)

class DQN():
    def __init__(self,env):
        self.epsilon_step = ( INIT_EPSILON - FINAL_EPSILON ) / 10000
        self.action_dim = env.action_space.n
        print( env.observation_space )
        self.state_dim = env.observation_space.shape[0]
        self.neuron_num = 100
        self.replay_buffer = deque()
        self.epsilon = INIT_EPSILON
        self.sess = tf.InteractiveSession()
        self.init_network()


        self.sess.run( tf.initialize_all_variables() )

    def init_network(self):
        self.input_layer = tf.placeholder( tf.float32, [ None, self.state_dim ] )
        self.action_input = tf.placeholder( tf.float32, [None, self.action_dim] )
        self.y_input = tf.placeholder( tf.float32, [None] )
        w1 = get_weights( [self.state_dim, self.neuron_num] )
        b1 = get_bias([self.neuron_num])

        hidden_layer = tf.nn.relu( tf.matmul( self.input_layer, w1 ) + b1 )

        w2 = get_weights( [ self.neuron_num, self.action_dim ] )
        b2 = get_bias( [ self.action_dim ] )


        self.Q_value = tf.matmul( hidden_layer, w2 ) + b2

        value = tf.reduce_sum( tf.multiply( self.Q_value, self.action_input ), reduction_indices = 1 )

        self.cost = tf.reduce_mean( tf.square( value - self.y_input ) )

        self.optimizer = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.cost)

        return

    def percieve(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros( [ self.action_dim ] )
        one_hot_action[ action ] = 1

        self.replay_buffer.append( [ state, one_hot_action, reward, next_state, done ] )

        if len( self.replay_buffer ) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        if len( self.replay_buffer ) > TRAIN_START_SIZE:
            self.train()

    def train(self):
        mini_batch = random.sample( self.replay_buffer, BATCH_SIZE )
        state_batch = [data[0] for data in mini_batch]
        action_batch = [data[1] for data in mini_batch]
        reward_batch = [ data[2] for data in mini_batch ]
        next_state_batch = [ data[3] for data in mini_batch ]
        done_batch = [ data[4] for data in mini_batch ]

        y_batch = []

        next_state_reward = self.Q_value.eval( feed_dict = { self.input_layer : next_state_batch } )


        for i in range( BATCH_SIZE ):
            if done_batch[ i ]:
                y_batch.append( reward_batch[ i ] )
            else:
                y_batch.append( reward_batch[ i ] + GAMMA * np.max( next_state_reward[i] ) )


        self.optimizer.run(
            feed_dict = {
                self.input_layer:state_batch,
                self.action_input:action_batch,
                self.y_input:y_batch
            }
        )

        return
    def get_greedy_action(self, state):
        value = self.Q_value.eval( feed_dict = { self.input_layer : [state] } )[ 0 ]
        return np.argmax( value )

    def get_action(self, state):
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= self.epsilon_step
        if random.random() < self.epsilon:
            return random.randint( 0, self.action_dim - 1 )
        else:
            return self.get_greedy_action(state)
def main():
    env = gym.make(ENV_NAME)
    agent = DQN( env )
    for episode in range(EPISDOE):
        total_reward = 0
        state = env.reset()
        for step in range(STEP):
            env.render()
            action = agent.get_action( state )
            next_state, reward, done, _ = env.step( action )
            total_reward += reward
            agent.percieve( state, action, reward, next_state, done )
            if done:
                break
            state = next_state

        print 'total reward this episode is: ', total_reward


if __name__ == "__main__":
    main()
