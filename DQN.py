
#
#  This script is to initialize and train a deep Q-network based on a given environment
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Env
import statistics
import Constant
import Network_Mock
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import time



# ---------------------------- hyper parameters ---------------------------

BATCH_SIZE = 32 # batch size for network training
LEARNING_RATE = 0.02 # learning rate
EPSILON = 0.9 # used for greedy policy
GAMMA = 0.9 # discount factor
TARGET_REPLACE_ITER = 100 # target network update frequency
MEMORY_CAPACITY = 2000 # experience buffer size
EPISODE_NUM = 50# number of episodes used to training a DQN


class Net(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        '''
        To initialize a neural network
        :param N_STATES: dimension of state vector (representation), output of the environment
        :param N_ACTIONS: dimension of action space
        '''

        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 20)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(20, 20)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(20, 20)
        self.fc3.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(20, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions_value = self.out(x)
        return actions_value




class DQN(object):

    def __init__(self, N_STATES, N_ACTIONS):
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LEARNING_RATE)
        self.loss_func = nn.MSELoss()
        self.n_states = int(N_STATES)
        self.n_actions = int(N_ACTIONS)

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action


    # the performance of evaluate (main) network may be unstable or overfitting,
    # In that case, we can avail the target network in test trials
    def choose_action_by_target_net(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            actions_value = self.target_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action



    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1



    def learn(self):
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1


        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states])

        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))

        b_r = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])

        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:])



        q_eval = self.eval_net(b_s).gather(1, b_a)

        q_next = self.target_net(b_s_).detach()

        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def flush_buffer(self):
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, self.n_states * 2 + 2))




def training_op(dqn, env, tx_load):


    env.reset()
    dqn.flush_buffer()

    success_tx_count = 0
    failed_tx_count = 0
    total_tx_count = 0


    while total_tx_count < tx_load:
        # the payment dataset is too large,
        # so control the size of transaction data used for training

        tx = random.choice(env.txs)
        # print(tx)

        total_tx_count += 1


        tx_, avalible_paths, avai_path_lens, avai_bottlenecks, avai_network_balance_qualities = env.shortest_path_searching_v2(
            tx)

        # if there are no available paths for this transaction, then discard it
        if len(avalible_paths) <= 0:
            # print('Warning: there are no available paths for transaction ' + str(tx))
            failed_tx_count += 1
            continue

        state = env.feature_vector_gen(tx_, avalible_paths, avai_path_lens, avai_bottlenecks,
                                       avai_network_balance_qualities)
        action = dqn.choose_action(state)  # 输入该步对应的状态s，选择动作
        state_, reward, succ_info = env.step(state, action)  # 执行动作、更新环境、反馈奖励

        if succ_info:
            success_tx_count += 1
        else:
            failed_tx_count += 1

        dqn.store_transition(state, action, reward, state_)

        if dqn.memory_counter > MEMORY_CAPACITY:

            dqn.learn()

    print('Success ratio : ' + str(round(success_tx_count / total_tx_count, 4)))
    print('Fail ratio : ' + str(round(failed_tx_count / total_tx_count, 4)))
    print('\n')



# def training():
#
#     topo = 'LN_topo' # either 'LN_topo' or 'ER_graph'
#     param_k = 4
#     tx_load = 40000
#     capacity_factor = 1
#
#     model_path = './data/model/topo_' + str(topo) + '_k' + str(param_k) + '_episode' + str(EPISODE_NUM) + '_dqn.pt'
#
#     Net = Network_Mock.load_network(topo=topo, capacity_factor=capacity_factor)
#     env = Env.Environment(**{'network': Net, 'param_k': param_k})
#
#     training_op(env, model_path, param_k, tx_load)
#
#     print('Training process finish. Parameters are as follows: \n'
#           'PCN topo = ' + str(topo) + ' | K = ' + str(param_k) + ' | capacity factor = ' + str(capacity_factor))
#


def tx_amp_op(env, dqn, tx):
    payer = int(tx[0])
    recipient = int(tx[1])
    value = int(tx[2])

    tx_unit = 100000
    tmp_graph = env.network_graph
    succ_flag = True
    info = 'AMP Succ'

    subTx_num = (value // tx_unit) +1

    for sub_TX_i in range(0, subTx_num):

        if sub_TX_i == subTx_num - 1:
            sub_tx = (payer, recipient, int(value % tx_unit))
        else:
            sub_tx = (payer, recipient, tx_unit)

        tx_, avalible_paths, avai_path_lens, avai_bottlenecks, avai_network_balance_qualities = env.shortest_path_searching_v2(sub_tx)

        if len(avalible_paths) <= 0:
            succ_flag = False
            info = 'No available path for sub-tx: ' + str(sub_tx)
            break

        state = env.feature_vector_gen(sub_tx, avalible_paths, avai_path_lens, avai_bottlenecks,
                                       avai_network_balance_qualities)


        action = dqn.choose_action_by_target_net(state)
        state_, reward, succ_info = env.step(state, action)

        if not succ_info:
            succ_flag = False
            info = 'Fail to execute sub-tx: ' + str(sub_tx)
            env.network_graph = tmp_graph

            break


    return succ_flag , info



def throughout_eval():

    topo = 'LN_topo' # either 'LN_topo' or 'ER_graph'
    param_k = 14
    tx_load = 2000
    model_path = './data/model/topo_' + str(topo) + '_k' + str(param_k) + '_episode' + str(EPISODE_NUM) + '_dqn_amp.pt'

    dqn = torch.load(model_path)  # load the trained model
    # load the payment channel network
    Net = Network_Mock.load_network(topo=topo, capacity_factor=25, node_nb=16083, prob=0.0006)
    # instantiate the environment
    env = Env.Environment(**{'network': Net, 'param_k': param_k})

    success_ratio_arr = []
    no_avai_path_tx_ratio_arr = []
    exec_failed_tx_ratio_arr = []

    trails = 5
    for i in range(trails):

        env.reset()

        success_tx_count = 0
        failed_tx_count = 0
        no_avai_path_tx_count = 0
        exec_failed_tx_count = 0
        total_tx_count = 0

        start_time = time.time()
        for tx in env.txs:


            if total_tx_count >= tx_load:
                break


            if int(tx[0]) not in env.network_graph:
                print('Warning: payer node : ' + str(tx[0]) + ' not exist')
                continue
            if int(tx[1]) not in env.network_graph:
                print('Warning: recipient node : ' + str(tx[1]) + ' not exist')
                continue

            total_tx_count += 1
            tx_, avalible_paths, avai_path_lens, avai_bottlenecks, avai_network_balance_qualities = env.shortest_path_searching_v2(tx)

            if len(avalible_paths) <= 0:
                # print("Warning: there are no available paths for transaction " + str(tx))
                failed_tx_count += 1
                no_avai_path_tx_count += 1
                continue

            state = env.feature_vector_gen(tx_, avalible_paths, avai_path_lens, avai_bottlenecks,
                                           avai_network_balance_qualities)

            # action = dqn.choose_action(state)
            action = dqn.choose_action_by_target_net(state)
            state_, reward, succ_info = env.step(state, action)

            if succ_info :
                success_tx_count += 1
            else:
                failed_tx_count += 1
                exec_failed_tx_count += 1

        duration = time.time() - start_time
        print('Trail ' + str(i) + ' finish, execution duration: ' + str(round(duration,2)) + ' s.')
        print('Success ratio : ' + str(round(success_tx_count / total_tx_count, 4)))
        print('Fail ratio : ' + str(round(failed_tx_count / total_tx_count, 4)))

        success_ratio_arr.append(round(success_tx_count / total_tx_count, 4))
        no_avai_path_tx_ratio_arr.append(round(no_avai_path_tx_count / total_tx_count, 4))
        exec_failed_tx_ratio_arr.append(round(exec_failed_tx_count / total_tx_count, 4))


    print('\nEvaluation process finish, topo = ' + str(topo) + ' , k = ' + str(param_k) + ' .')
    average_succ_ratio = round(np.mean(success_ratio_arr), 4)
    no_avai_path_proportion_resl =  round(np.mean(no_avai_path_tx_ratio_arr),4)
    exe_failed_proportion_resl =  round(np.mean(exec_failed_tx_ratio_arr),4)
    # print('average success ratio is: ' + str(average_succ_ratio))
    # print('failed tx ratio (due to no available paths) : ' + str(no_avai_path_proportion_resl))
    # print('failed tx ratio (there are paths but fail to execute) : ' + str(exe_failed_proportion_resl))



def train_in_LN():


    topo = 'LN_topo' # either 'LN_topo' or 'ER_graph'
    # param_k = 4
    tx_load = 20000

    train_start_time = time.time()
    # param_k_arr = range(10, 20)
    param_k_arr = [7]

    for param_k in param_k_arr:

        N_ACTIONS = param_k  # size of action space

        # dimension of state vector (representation) in the environment
        # the dimension is determined by Env.feature_vector_gen
        # 3 denotes the dimension of transaction
        # 3*(Constant.param_k) denotes that 3 features of each path multiply param_k paths
        N_STATES = 3 + 3 * (param_k)

        dqn = DQN(N_STATES, N_ACTIONS)

        for i in range(EPISODE_NUM):  # episode

            for capacity_factor in [1,5,15,20,25]:
                print('Training start, Params : '
                      'PCN topo = ' + str(topo) + ' | K = ' + str(param_k) + ' | episode = ' + str(i+1) + ' | capacity factor = ' + str(
                    capacity_factor))

                Net = Network_Mock.load_network(topo=topo, capacity_factor=capacity_factor)
                env = Env.Environment(**{'network': Net, 'param_k': param_k})

                training_op(dqn, env, tx_load)


        model_path = './data/model/topo_' + str(topo) + '_k' + str(param_k) + '_episode' + str(
                    EPISODE_NUM) + '_version3.pt'
        torch.save(dqn, model_path)  # save the trained model to the given path

    train_duration = time.time() - train_start_time
    print('Training finish, training duration : ' + str(round(train_duration, 2)) + ' s.\n')


def eval_in_LN():


    topo = 'LN_topo'  # either 'LN_topo' or 'ER_graph'
    tx_load = 30000

    eval_res_file = 'result/k_14_RewardNormal_version3.txt'
    process_result = 'result/k_14_RewardNormal_version3_process.txt'


    with open(eval_res_file, 'w', encoding='utf-8') as f:
        f.write('param k | capacity factor | succ ratio | unavai_path ratio | exe_fail ratio| average_balanced\n')

    eval_start_time = time.time()
    param_k_arr = [14]
    for param_k in param_k_arr:
        for capacity_factor in range(1,2):

            eval_1_start_time = time.time()


            model_path = './data/model/topo_' + 'LN_topo' + '_k' + str(param_k) + '_episode' + str(
                   EPISODE_NUM) + 'version3.pt'

            dqn = torch.load(model_path)  # load the trained model
            # load the payment channel network
            Net = Network_Mock.load_network(topo=topo, capacity_factor=capacity_factor)
            # instantiate the environment
            env = Env.Environment(**{'network': Net, 'param_k': param_k})

            success_ratio_arr = []
            no_avai_path_tx_ratio_arr = []
            exec_failed_tx_ratio_arr = []
            bal_quality_arr = []

            trails =10
            for i in range(trails):

                # print('\nTrail %s is starting.' % i)
                env.reset()

                success_tx_count = 0
                failed_tx_count = 0
                no_avai_path_tx_count = 0
                exec_failed_tx_count = 0
                total_tx_count = 0

                for tx in env.txs:

                    #print('In Trail ' + str(i) + ' , Tx executed : ' + str(100 * round(total_tx_count/tx_load, 4)) + '%')

                    # control the size of txs used for training since the payment dataset is too large
                    if total_tx_count >= tx_load:
                        break


                    total_tx_count += 1

                    # print(total_tx_count)

                    tx_, avalible_paths, avai_path_lens, avai_bottlenecks, avai_network_balance_qualities = env.shortest_path_searching_v2(tx)

                    if len(avalible_paths) <= 0:
                        # print("Warning: there are no available paths for transaction " + str(tx))
                        # use Amp to devide this transaction to multiple sub-tx
                        succ_flag, info = tx_amp_op(env, dqn, tx)
                        if succ_flag:
                            success_tx_count += 1
                            # print('tx : ' + str(tx) + ' AMP succ')
                        else:
                            failed_tx_count += 1
                            no_avai_path_tx_count += 1
                            # print('tx : ' + str(tx) + ' AMP fail. info : ')
                            # print(info)
                        continue


                    state = env.feature_vector_gen(tx_, avalible_paths, avai_path_lens, avai_bottlenecks,
                                                   avai_network_balance_qualities)

                    # action = dqn.choose_action(state)
                    action = dqn.choose_action_by_target_net(state)
                    state_, reward, succ_info = env.step(state, action)

                    if succ_info:
                        success_tx_count += 1
                    else:
                        failed_tx_count += 1
                        exec_failed_tx_count += 1
                        print('tx : ' + str(tx) + ' fail to exe. Chosen action : ' + str(action) + ' Reward : ' +str(reward))




                    if total_tx_count % 1000 == 0:

                        bal_qua = env.balance_quality()

                        with open(process_result, 'a', encoding='utf-8') as f:
                            f.write(str(param_k) + ' | ' + str(capacity_factor) + ' | '
                                    + str(round(success_tx_count / total_tx_count, 4)) + ' | '
                                    + str(round(no_avai_path_tx_count / total_tx_count, 4)) + ' | '
                                    # + str(round(exec_failed_tx_count / total_tx_count, 4)) + ' | '
                                    + str(round(exec_failed_tx_count / total_tx_count, 4)) + ' | '
                                    + str(bal_qua) + '\n')


                # print('Success ratio : ' + str(round(success_tx_count / total_tx_count, 4)))
                # print('Fail ratio : ' + str(round(failed_tx_count / total_tx_count, 4)))

                success_ratio_arr.append(round(success_tx_count / total_tx_count, 4))
                no_avai_path_tx_ratio_arr.append(round(no_avai_path_tx_count / total_tx_count, 4))
                exec_failed_tx_ratio_arr.append(round(exec_failed_tx_count / total_tx_count, 4))
                bal_quality_arr.append(env.balance_quality())


            # print('\nEvaluation process finish, topo = ' + str(topo) + ' , k = ' + str(param_k) + ' .')
            average_succ_ratio = round(np.mean(success_ratio_arr), 4)
            no_avai_path_proportion_resl = round(np.mean(no_avai_path_tx_ratio_arr), 4)
            exe_failed_proportion_resl = round(np.mean(exec_failed_tx_ratio_arr), 4)
            average_net_bal_quality = round(np.mean(bal_quality_arr), 4)


            with open(eval_res_file, 'a', encoding='utf-8') as f:
                f.write(str(param_k) + ' | ' + str(capacity_factor) + ' | '
                        + str(average_succ_ratio) + ' | '
                        + str(no_avai_path_proportion_resl) + ' | '
                        + str(exe_failed_proportion_resl) + ' | '
                        + str(average_net_bal_quality) + '\n')

            eval_1_duration = time.time() - eval_1_start_time
            print('Eval: K = ' + str(param_k) + ' | capacity factor = ' + str(capacity_factor) + ' , Time: ' + str(round(eval_1_duration,2)) + ' s')


    eval_duration = time.time() - eval_start_time

    print('\n\nEvaluation finish, evaluation duration : ' + str(round(eval_duration, 2)) + ' s.')


if __name__ == '__main__':

    train_in_LN()
    # eval_in_LN()

