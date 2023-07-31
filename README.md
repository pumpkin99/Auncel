# The Implementation of Auncel

This is an implementation of the paper *Auncel: Imbalance Prevention in Payment Channel Networks with Deep Reinforcement Learning*.

## Basic Usage

Run `DQN.py` to train a deep Q-network model with the given environment and transaction data. Specifically, you should first initialize a DQN instance and create (or load) a PCN instance according to the specified topology (Lightning Network or Erdos-Renyi graph),  which is utilized to initialize a PCN environment.  The DQN instance and PCN environment together with the given transaction dataset will be transferred to the function `training_op()` to perform the training operation. When the training process finishes, the DQN model will be saved to a specified path. Next, you can use a transaction dataset to test the resulting model to evaluate its performance on various metrics.

Additionally, you may want to try out different configurations or other PCN topologies. If so, you can rework the scripts  `Network_Mock.py` and `Env.py`, and then reinitialize a PCN environment.

## Scripts Description:

**1) Network_Mock.py:** This script is used to generate a PCN instance with a given topology via the NetworkX toolkit. It contains three functions whose functions are as follows:

- `network_gen_ln_topo(capacity_factor):` To generate a payment channel network by Lighting Network topology with a given capacity factor. The topology comes from the snapshot of LN topology on 2021-03-31 (from publication [dblp: Shaduf: Non-Cycle Payment Channel Rebalancing](https://dblp.org/rec/conf/ndss/GeZ0G22.html)), which contains 10,529 nodes and 38,910 channels.

- `network_gen_er_graph_v2(node_nb=16083, prob=0.0006):` To simulate a payment channel network with the default number of nodes and connectivity by Erdos Renyi Random Graph. You can modify parameters `node_nb` and `prob` to control the number of nodes and density of edges to suit specific scenarios.

- `load_network():`  To load a generated network from the local file. You should specify a network topology: either 'LN_topo' or 'ER_graph', where 'LN_topo' denotes the network generated by the lighting network, and 'ER_graph' denotes the network generated by Erdos Renyi Random Graph.

**2) Env.py:** This script is an implementation of the payment channel network environment based on the network instance generated by script `Network_Mock.py`. It defines a class `Environment ` and contains the following main functions:

- `txs_generate():`To extract transaction data from a given dataset and organize it into a proper structure.

- `step(state, action):` To execute a transaction based on the given path, and then update the payment channel network. Parameter `state` represents the current state of payment channel network, and `action` denotes a routing path recommended by our DQN model.

- `reset():` To reset the PCN environment to restore the initialization state for a new round of training or testing.

**3) DQN.py:** This script is used to initialize and train a deep Q-network based on a given PCN environment. You can invoke the function `training_op()` to train an initialized DQN model with the given PCN environment. The class `DQN` is implemented by the pytorch toolkit to initialize a DQN model. In this class, the following functions need attention:

- `choose_action(s):` To choose an optimal action based on state s, in our context, that is, recommend the optimal path based on the PCN state s.

- `store_transition():` Record information to the experience buffer serves the experience replay mechanism.

- `learn():`Fetch a batch of data from the experience buffer for learning, and update the parameters of DQN. 

**4) Constant.py:** The definition of constants, such as weights for each objective in Auncel's multi-objective decision.