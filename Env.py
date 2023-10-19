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

        # Bitcoin transaction dataset
        # payment_value_file = './data/dataset/payment_value_satoshi_03.csv'
        payment_value_file = './data/dataset/payment_value_satoshi.csv'

        # Since the transaction amounts in the Bitcoin dataset are much larger than PCN,
        # a threshold is set for the loaded payment amount
        # e.g., the median of channel capacity as Shaduf did
        payment_value_threshold = self.capacity_median
        # payment_value_threshold = None # do not set a threshold

        node_zone = list(self.network_graph.nodes()) # all nodes
        txs = []
        with open(payment_value_file) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                payer, recipient, value = 0, 0, 0

                while True:
                    # sample a payer or payee uniformly from all node as Shaduf did
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
            # If the suggested path is unavailable,
            # a large negative feedback is returned to prompt DQN to recommend available paths.
            reward = -100000000
            succ_info = False

        else:
            path = self.candidate_paths[action]
            path_capacity_sum = 0

            # --------- calculate the quality or indicator before executing the tx
            network_balance_quality_old = 0  # used for evaluating the quality of network balance
            for i in range(0, len(path)):
                edge = path[i]
                c_uv = self.network_graph.edges[edge[0], edge[1]]['weight']
                c_vu = self.network_graph.edges[edge[1], edge[0]]['weight']
                network_balance_quality_old += abs(c_uv - c_vu)
                path_capacity_sum += c_uv

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

            # -------- give the updated state ---------
            state_[4+(3*action)] -= tx_value
            state_[5+(3*action)] = network_balance_quality_new
            succ_info = True

        #  reset the candidate paths saved for current transaction
        self.candidate_paths = []

        return state_, reward, succ_info


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

            # solve the bottleneck for this path
            # Initialized by a large value, then find the minimum capacity over this path
            bottleneck = 100000000000
            # calculate the network balance quality for this path
            network_balance_quality = 0  # used for evaluating the quality of network balance

            for edge in path:
                # get the capacity of channel (i.e., the weight of edge)
                capacity = self.network_graph.get_edge_data(edge[0], edge[1])['weight']

                c_uv = self.network_graph.edges[edge[0], edge[1]]['weight']
                c_vu = self.network_graph.edges[edge[1], edge[0]]['weight']
                network_balance_quality += abs(c_uv - c_vu)

                if capacity < bottleneck:
                    # find the minimum capacity over this path
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


        return tx, avalible_paths, avai_path_lens, avai_bottlenecks, avai_network_balance_qualities



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

            self.candidate_paths.append(avalible_paths[i])
            state.append(avai_path_lens[i])
            state.append(avai_bottlenecks[i])
            state.append(avai_network_balance_qualities[i])


        if len(avalible_paths) < self.param_k:
            # if the number of available paths is less than k

            # fill the state with meaningless value, e.g., -1
            for i in range(0, self.param_k - len(avalible_paths)):

                state.append(-1)
                state.append(-1)
                state.append(-1)

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

        return net_balance_quality


