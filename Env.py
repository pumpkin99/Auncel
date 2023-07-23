#
# This script is an implementation of the payment channel network
#
import csv
import numpy as np
import random
import Constant
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import Network_Mock
import time
import more_itertools


class Environment():

    def __init__(self, **kwargs):

        # self.topo = kwargs['topo'] # 'topo' specify a network topology: is either 'LN_topo' or 'ER_graph'
        self.original_net = kwargs['network'] # original network (will not update, used to reset the env )
        self.network_graph = kwargs['network'] # payment channel network (will update when executing transactions)
        self.node_nb = int(len(self.network_graph.nodes)) # the number of nodes in pcn
        self.channel_nb = int(len(self.network_graph.edges) / 2) # the number of opened channels in pcn. Each channel contains 2 opposite edges
        # the median and mean of channel capacities
        self.capacity_median, self.capacity_mean = self.net_statistic()

        self.param_k = kwargs['param_k']  # the maximum number of candidate paths input to DQN

        self.txs = self.txs_generate() # a list stores all payment transactions
        self.candidate_paths = []


    def net_statistic(self):
        '''
        some statistic of network
        :return:
        '''
        capacities = []
        for edge in self.network_graph.edges:
            u, v = edge[0], edge[1]
            capacity = self.network_graph.get_edge_data(u, v)['weight']
            capacities.append(capacity)

        # the median and mean of channel capacities
        capacity_median = np.median(capacities)
        capacity_mean = np.mean(capacities)
        return capacity_median, capacity_mean


    def txs_generate(self):

        # Bitcoin transaction value
        # payment_value_file = './data/dataset/Reward_Transaction.csv'
        # payment_value_file = './data/dataset/payment_value_satoshi_03.csv'
        payment_value_file = './data/dataset/payment_value_satoshi.csv'

        # payment_value_file = './data/dataset/test/test1.csv'
        # the threshold for payment value, i.e., the median of channel capacity
        # payment_value_threshold = self.capacity_median
        payment_value_threshold = None
        # payment_value_threshold = 466359
        # print('payment value threshold : ' + str(payment_value_threshold))
        # print('capacity median : ' + str(self.capacity_median))
        # print('capacity mean : ' + str(self.capacity_mean))

        node_zone = list(self.network_graph.nodes()) # all nodes
        txs = []
        with open(payment_value_file) as csvfile:
            csv_reader = csv.reader(csvfile)
            # header = next(csv_reader)
            for row in csv_reader:

                payer, recipient, value = 0, 0, 0

                while True: # sample a payer or payee uniformly from all node
                    payer = int(random.choice(node_zone))
                    recipient = int(random.choice(node_zone))
                    if payer != recipient:
                        break

                v = int(float(row[0]))
                if payment_value_threshold == None:
                    if 0 < v:
                        value = v
                        txs.append((payer, recipient, value))
                else:
                    if 0 < v < payment_value_threshold:
                        value = v
                        txs.append((payer, recipient, value))
        return txs


    def reset(self):
        '''
        To reset the environment
        :return:
        '''

        self.network_graph = self.original_net # reload the graph
        self.capacity_median, self.capacity_mean = self.net_statistic()
        self.txs = self.txs_generate()  # a list stores all payment transactions
        self.candidate_paths = []



    def step(self, state, action):
        '''
        execute the tx based on the given path, then update the payment channel network

        :param state: the transaction to be executed
        :param action: a path suggested by the DQN
        :return: state_, reward, done,
                state_ : updated state
                reward : instant reward for agent
                succ_info : flag (weather this Tx success)
        '''

        state_ = state
        reward = 0 # reward to be returned
        succ_info = False  # flag denotes whether this tx success or not

        tx_payer, tx_payee, tx_value = state[0], state[1], state[2]

        # check the path is available or not
        if action >= len(self.candidate_paths):
            # if the given path is unavailable, then choose the first available path as default
            # path = self.candidate_paths[0]
            reward = -100000000
            succ_info = False

        else:
            path = self.candidate_paths[action]
            path_capacity_sum = 0 # 统计交易执行前，本条路径上，所有通道容量的和

            # --------- calculate the quality or indicator before executing the tx
            network_balance_quality_old = 0  # used for evaluating the quality of network balance
            for i in range(0, len(path)):
                edge = path[i]
                c_uv = self.network_graph.edges[edge[0], edge[1]]['weight']
                c_vu = self.network_graph.edges[edge[1], edge[0]]['weight']
                network_balance_quality_old += abs(c_uv - c_vu)
                path_capacity_sum += c_uv # 统计交易执行前，本条路径上，所有通道容量的和

            # ------ update the network graph, especially the weight of edges in path -------
            for i in range(0, len(path)):
                edge = path[i]
                self.network_graph.edges[edge[0], edge[1]]['weight'] -= tx_value
                self.network_graph.edges[edge[1], edge[0]]['weight'] += tx_value


            # --------- caculate the reward (utility) for some indicator such as the quality of network balance
            # after transaction execution
            network_balance_quality_new = 0  # used for evaluating the quality of network balance
            for i in range(0, len(path)):
                edge = path[i]
                c_uv = self.network_graph.edges[edge[0], edge[1]]['weight']
                c_vu = self.network_graph.edges[edge[1], edge[0]]['weight']
                network_balance_quality_new += abs(c_uv - c_vu)





            # -------- Summarize the reward with above indicator of quality -------


            network_balance_quality_old_normal = (network_balance_quality_old - 10000)/9989999
            network_balance_quality_new_normal = (network_balance_quality_new - 10000)/9989999
            network_balance_quality =1 - abs((network_balance_quality_old_normal) - (network_balance_quality_new_normal))


            reward_balance = network_balance_quality * Constant.reward_balance_factor


            path_cost = (len(path)-1)/ 9

            reward_time = -(path_cost*Constant.htlc_locked_time_delta*Constant.reward_time_factor)




            reward_fee = -(Constant.k_fee**2 + Constant.gamma_fee*(path_capacity_sum/100000000))*Constant.reward_fee_factor
            reward = reward + reward_balance + reward_time + reward_fee
            # print('reward_All   ' + str(round(reward, 4)))

            # -------- give the updated state ---------
            state_[4+(3*action)] -= tx_value # 更新bottleneck
            state_[5+(3*action)] = network_balance_quality_new # 更新均衡度

            succ_info = True

        #  reset the candidate paths saved for current transaction
        self.candidate_paths = []

        return state_, reward, succ_info



    # def shortest_path_searching(self, tx):
    #     '''
    #     searching k shortest paths for payment tx in current network state.
    #     '''
    #     k = self.param_k
    #
    #     payer = int(tx[0])
    #     recipient = int(tx[1])
    #     value = int(tx[2])
    #
    #     avalible_paths = []  # stored the available paths
    #     avai_bottlenecks = []  # bottleneck for available paths
    #     avai_path_lens = []  # length (edge number) of paths
    #     avai_network_balance_qualities = []  # store network balance quality for each available path
    #
    #     count = 0 # count the number of found path
    #     # temporarily store the checked paths and weights of edges in checked path
    #     checked_paths = []
    #     checked_paths_reverse = []
    #     checked_paths_weight = []
    #     checked_paths_weight_reverse = []
    #
    #     loop_count = 0
    #     while True:
    #         # to avoid too numbers of executing this loop
    #         if loop_count >= 1000 * k:
    #             break
    #         loop_count += 1
    #
    #         # if there is no path between {source} and {target},
    #         # nx.shortest_path will pose an exception:"NetworkXNoPath"
    #         if not nx.has_path(self.network_graph, payer, recipient):
    #             break
    #
    #         shortest_path = nx.shortest_path(self.network_graph, payer, recipient)
    #
    #         # convert the path (a list of nodes) to edge path (a list of edges)
    #         edge_path = []
    #         for i in range(0, len(shortest_path) - 1):
    #             edge = (int(shortest_path[i]), int(shortest_path[i + 1]))
    #             edge_path.append(edge)
    #
    #         # check this path whether or not available
    #         avai = True  # flag Indicates whether the path is available
    #         bottleneck = 100000000000  # solve the bottleneck for this path
    #         # calculate the network balance quality for this path
    #         network_balance_quality = 0  # used for evaluating the quality of network balance
    #
    #         for edge in edge_path:
    #             # get the capacity of channel (i.e., the weight of edge)
    #             capacity = self.network_graph.get_edge_data(edge[0], edge[1])['weight']
    #
    #             c_uv = self.network_graph.edges[edge[0], edge[1]]['weight']
    #             c_vu = self.network_graph.edges[edge[1], edge[0]]['weight']
    #             network_balance_quality += abs(c_uv - c_vu)
    #
    #             if capacity < bottleneck:
    #                 bottleneck = capacity
    #
    #             if value > capacity:  # the value of tx is larger than channel capacity
    #                 avai = False  # this path is unavailable
    #
    #         if avai:  # this path is available for transaction tx
    #             avalible_paths.append(edge_path)
    #             avai_path_lens.append(len(edge_path))
    #             avai_bottlenecks.append(bottleneck)
    #             avai_network_balance_qualities.append(network_balance_quality)
    #             count += 1
    #
    #         # temporarily store the path
    #         weight_of_path = []
    #         weight_of_path_reverse = [] # weight of reverse direction
    #         edge_path_reverse = [] # the reverse direction of edge_path
    #         for edge in edge_path:
    #             weight = self.network_graph.get_edge_data(edge[0], edge[1])['weight']
    #             weight_of_path.append(weight)
    #
    #             weight_reverse = self.network_graph.get_edge_data(edge[1], edge[0])['weight']
    #             weight_of_path_reverse.append(weight_reverse)
    #
    #             edge_reverse = (edge[1], edge[0]) # the reverse direction of this edge
    #             edge_path_reverse.append(edge_reverse) # the reverse direction of edge_path
    #
    #
    #         checked_paths_weight.append(weight_of_path)
    #         checked_paths_weight_reverse.append(weight_of_path_reverse)
    #         checked_paths.append(edge_path)
    #         checked_paths_reverse.append(edge_path_reverse) # store the diverse direction of path
    #
    #         # temporarily remove this path
    #         self.network_graph.remove_edges_from(edge_path)
    #         self.network_graph.remove_edges_from(edge_path_reverse)
    #
    #         if count >= k:
    #             break
    #
    #
    #     # recover the network by add the removed path before
    #     for i in range(0, len(checked_paths)):
    #         # path consist of edges
    #         edge_path = checked_paths[i]
    #         # corresponding weight for each edge of this path
    #         weight_of_path = checked_paths_weight[i]
    #
    #         # the reversed direction of path
    #         edge_path_reverse = checked_paths_reverse[i]
    #         weight_of_path_reverse = checked_paths_weight_reverse[i]
    #
    #         for j in range(0, len(edge_path)):
    #
    #             self.network_graph.add_edge(edge_path[j][0], edge_path[j][1], weight= weight_of_path[j])
    #             self.network_graph.add_edge(edge_path_reverse[j][0], edge_path_reverse[j][1], weight= weight_of_path_reverse[j])
    #
    #     # print('\nfor tx : ' + str(tx) + ', available paths are ' + str(len(avalible_paths)))
    #
    #     return tx, avalible_paths, avai_path_lens, avai_bottlenecks, avai_network_balance_qualities



    def shortest_path_searching_v2(self, tx):
        '''
        searching k shortest paths for payment tx in current network state.
        '''
        k = self.param_k
        payer = int(tx[0])
        recipient = int(tx[1])
        value = int(tx[2])

        avalible_paths = []  # stored the available paths
        avai_bottlenecks = []  # bottleneck for available paths
        avai_path_lens = []  # length (edge number) of paths
        avai_network_balance_qualities = []  # store network balance quality for each available path

        shortest_paths = nx.shortest_simple_paths(self.network_graph, payer, recipient)

        # print(time.time())

        check_count = 0
        count = 0
        for p in shortest_paths:

            # convert the path (a list of nodes) to edge path (a list of edges)
            path = []
            for i in range(0, len(p) - 1):
                edge = (int(p[i]), int(p[i + 1]))
                path.append(edge)

            # check this path whether or not available
            avai = True  # flag Indicates whether the path is available
            bottleneck = 100000000000  # solve the bottleneck for this path
            # calculate the network balance quality for this path
            network_balance_quality = 0  # used for evaluating the quality of network balance

            for edge in path:
                # get the capacity of channel (i.e., the weight of edge)
                capacity = self.network_graph.get_edge_data(edge[0], edge[1])['weight']

                c_uv = self.network_graph.edges[edge[0], edge[1]]['weight']
                c_vu = self.network_graph.edges[edge[1], edge[0]]['weight']
                network_balance_quality += abs(c_uv - c_vu)

                if capacity < bottleneck:
                    bottleneck = capacity

                if value > capacity:  # the value of tx is larger than channel capacity
                    avai = False  # this path is unavailable

            if avai:  # this path is available for transaction tx
                avalible_paths.append(path)
                avai_path_lens.append(len(path))
                avai_bottlenecks.append(bottleneck)
                avai_network_balance_qualities.append(network_balance_quality)
                count += 1

            if count >= k:
                break

            check_count += 1
            if check_count > k * 20:
                break

        # print(time.time())

        # print('Tx : ' + str(tx))
        # # print(more_itertools.ilen(shortest_paths))
        # print(avalible_paths)

        return tx, avalible_paths, avai_path_lens, avai_bottlenecks, avai_network_balance_qualities




    # def simple_path_searching(self, tx):
    #     '''
    #     searching all available paths for payment tx in current network state.
    #     :param tx: a dictionary which stores the info of payment tx to be executed
    #     :return: the tx and available paths
    #     '''
    #
    #
    #     payer = int(tx[0])
    #     recipient = int(tx[1])
    #     value = int(tx[2])
    #
    #     # print('\n For tx: ' + str(tx))
    #
    #     bool = nx.has_path(self.network_graph, payer, recipient)
    #
    #     if bool == False:
    #         # there is no path between payer and recipient
    #         pass
    #
    #     # get all simple paths
    #     simple_paths = nx.all_simple_edge_paths(self.network_graph, payer, recipient)
    #
    #     # print('simple path:')
    #     # print(simple_paths)
    #
    #     avalible_paths = [] # stored the available paths
    #     avai_bottlenecks = [] # bottleneck for available paths
    #     avai_path_lens = [] # length (edge number) of paths
    #     avai_network_balance_qualities = [] # store network balance quality for each available path
    #     for path in sorted(simple_paths):
    #         # check this path whether or not available
    #
    #         avai = True # flag Indicates whether the path is available
    #         bottleneck = 1000 # solve the bottleneck for this path
    #
    #         # calculate the network balance quality for this path
    #         network_balance_quality = 0  # used for evaluating the quality of network balance
    #
    #         for edge in path:
    #             # get the capacity of channel (i.e., the weight of edge)
    #             capacity = self.network_graph.get_edge_data(edge[0], edge[1])['weight']
    #
    #             c_uv = self.network_graph.edges[edge[0], edge[1]]['weight']
    #             c_vu = self.network_graph.edges[edge[1], edge[0]]['weight']
    #             network_balance_quality += abs(c_uv - c_vu)
    #
    #             if capacity < bottleneck:
    #                 bottleneck = capacity
    #
    #             if value > capacity: # the value of tx is larger than channel capacity
    #                 avai = False  # this path is unavailable
    #
    #         if avai : # this path is available for transaction tx
    #             avalible_paths.append(path)
    #             avai_path_lens.append(len(path))
    #             avai_bottlenecks.append(bottleneck)
    #             avai_network_balance_qualities.append(network_balance_quality)
    #
    #     return tx, avalible_paths, avai_path_lens, avai_bottlenecks, avai_network_balance_qualities



    def feature_vector_gen(self, tx_, avalible_paths, avai_path_lens, avai_bottlenecks, avai_network_balance_qualities):
        '''
        generate the state vector which is fed to DQN
        :param tx_: transaction
        :param avalible_paths: all available paths returned by function path_searching()
        :param avai_path_lens: len for each available path in 'avalible_paths' (returned by function path_searching())
        :param avai_bottlenecks: bottleneck for each available path in 'avalible_paths' (returned by function path_searching())
        :param avai_network_balance_qualities: balance quality for each available path in 'avalible_paths' (returned by function path_searching())
        :return: a state vector
        '''

        state = [int(tx_[0]), int(tx_[1]), int(tx_[2])]

        # select k paths to construct state vector
        for i in range(0,len(avalible_paths)):

            if i >= self.param_k: # reach the max number
                break

            self.candidate_paths.append(avalible_paths[i]) # 添加至候选路径
            state.append(avai_path_lens[i]) # 该路径的长度
            state.append(avai_bottlenecks[i]) # 该路径的bottleneck
            state.append(avai_network_balance_qualities[i]) # 该路径的均衡度


        if len(avalible_paths) < self.param_k: # if the number of available paths is less than k
            # fill the state with meaningless value, e.g., -1
            for i in range(0, self.param_k - len(avalible_paths)):

                state.append(-1)
                state.append(-1)
                state.append(-1)

        # print('state for tx:' + str(tx_))
        # print(state)

        return state



    def balance_quality(self):
        '''
        Returns: the balance quality of current network
        '''


        channel_num = self.network_graph.number_of_edges()

        bal_quality_sum = 0

        for edge in self.network_graph.edges():

            c_uv = self.network_graph.edges[edge[0], edge[1]]['weight']
            c_vu = self.network_graph.edges[edge[1], edge[0]]['weight']
            balance_quality = abs(c_uv - c_vu)
            bal_quality_sum += balance_quality

        net_balance_quality = round(bal_quality_sum/channel_num, 4)
        # print(net_balance_quality)

        return net_balance_quality


def main():
    # load the network
    # Param 'topo' specify a network topology: is either 'LN_topo' or 'ER_graph',
    # param 'capacity_factor' is only valid when topo='LN_topo'
    # params 'node_nb' and 'prob' are valid only when topo='ER_graph'

    Net = Network_Mock.load_network(topo='LN_topo', capacity_factor=1, node_nb=16083, prob=0.0006)


    param_k = 4

    env = Environment(**{'network':Net, 'param_k':param_k})
    env.balance_quality()
    #
    # count = 0
    # for tx in env.txs:
    #
    #     if int(tx[0]) not in env.network_graph:
    #         print('Warning: payer node : ' + str(tx[0]) + ' is not exist')
    #         continue
    #     if int(tx[1]) not in env.network_graph:
    #         print('Warning: recipient node : ' + str(tx[1]) + ' is not exist')
    #         continue
    #
    #     tx_, avalible_paths, avai_path_lens, avai_bottlenecks, avai_network_balance_qualities = env.shortest_path_searching(tx)
    #     # if there are no available paths for this transaction, then discard it
    #     if len(avalible_paths) <= 0:
    #         print('Warning: there are no available paths for transaction ' + str(tx))
    #         continue
    #     state = env.feature_vector_gen(tx_, avalible_paths, avai_path_lens, avai_bottlenecks, avai_network_balance_qualities)
    #     print('state whether changes')
    #     print(state)
    #     state_, reward, succ_info = env.step(state, 0)
    #     print(state_)
    #     print(reward)
    #     print(succ_info)
    #     count += 1
    #
    #     if count == 10:
    #         break





# def remove_test():
#
#     Net = Network_Mock.load_network(topo='LN_topo', capacity_factor=1, node_nb=16083, prob=0.0006)
#
#
#
#     shortest_path = nx.shortest_path(Net, 2489, 3655)
#
#     # convert the path (a list of nodes) to edge path (a list of edges)
#     edge_path = []
#     for i in range(0, len(shortest_path) - 1):
#         edge = (int(shortest_path[i]), int(shortest_path[i + 1]))
#         edge_path.append(edge)
#
#     checked_paths = []
#     checked_paths_weight = []
#
#
#     # temporarily store the path
#     checked_paths.append(edge_path)
#     weight_of_path = []
#     for edge in edge_path:
#         weight = Net.get_edge_data(edge[0], edge[1])['weight']
#         weight_of_path.append(weight)
#     checked_paths_weight.append(weight_of_path)
#
#     # temporarily remove this path
#     Net.remove_edges_from(edge_path)
#
#
#     # recover the network by add the removed path before
#     for i in range(0, len(checked_paths)):
#         # path consist of edges
#         edge_path = checked_paths[i]
#         # corresponding weight for each edge of this path
#         weight_of_path = checked_paths_weight[i]
#
#         for j in range(0, len(edge_path)):
#             u = edge_path[j][0]
#             v = edge_path[j][1]
#             weight = weight_of_path[j]
#             Net.add_edge(u, v, weight=weight)






if __name__ == '__main__':
    main()



