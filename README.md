# Project structure

```
├── agents
│   ├── dqn_agent # trained dqn agent (project solution)
│   │   ├── dqn_agent.py
│   │   ├── dqn_model.pth
│   │   ├── dqn_model.py
│   │   └── train_dqn.py
│   ├── naive_agent # naive baseline
│   │   └── my_rl_agent.py
│   └── q_learning_agent # q-learning baseline
│       ├── q_learning_agent.py
│       ├── q_table.pkl
│       └── train_q_learning.py
├── results # results used in report
│   ├── ...
├── README.md
├── evaluate_agents.py # evaluate dqn_agent against baselines
├── reinforcement_task.py # provided - contains evaluate_agent() function
├── requirements.txt
└── setup.sh

```


# How to run
For setup of virtual environment, train DQN agent and evaluate all models run ```setup.sh ``` by using:
```bash
bash setup.sh
```
or
```bash
./setup.sh
```
Make sure the script is executable: ```chmod +x setup.sh``` (shouldn't be necessary, but it's good practice).

Or create the venv manually:
```bash
...
```

For optional re-training of the models run from the project root:
* Q-learning agent:
    ```bash
    python -m agents.q_learning_agent.train_q_learning
    ```
* DQN agent:
    ```bash
    python -m agents.dqn_agent.train_dqn
    ```

To evaluate agents run:
```bash
python -m evaluate_agents
```

# Model Architecture

## Classic DQN
Deep Q-Network (DQN) [(Mnih 2014)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) is a reinforcement learning algorithm that uses a deep neural network to learn a Q-function, which is a function that estimates the expected return for taking a given action in a given state. The goal of the DQN algorithm is to learn a policy that maximizes the expected return by learning the Q-function and selecting the action with the highest estimated return in each state.

The DQN algorithm consists of two main components: a Q-network and an experience buffer. The Q-network is a deep neural network that takes in a state as input and outputs the estimated Q-values for each possible action. The experience buffer is a data structure that stores a set of experiences. The DQN algorithm works by interacting with the environment and storing the experiences in the experience buffer. The Q-network is then trained using a mini-batch of experiences uniformly sampled from the experience buffer. This process is known as experience replay and is used to decorrelate the experiences and to stabilize the learning process. The Q-network is updated using the loss function:

$$
\mathcal{L}_{\theta} = \frac{1}{B} \sum_{i=1}^{B} \bigl( \mathrm{TD}~(s_i, a_i, s^{'}_{i}) \bigr)^{2}
$$

With:

$$
\mathrm{TD}~(s_i, a_i, s^{'}_{i}) = Q_{\theta}~(s_i,a_i) - \bigl(r_{(s_i,a_i,s_{i}^{'})} + \gamma ~ \underset{a^{'}_{i} \sim \bar{Q}_{\theta}}{\mathrm{max}} ~ \bar{Q}_{\theta}~(s_{i}^{'},a_{i}^{'}) \bigr)
$$

Where $Q_{\theta}$ and $\bar{Q}_{\theta}$ denote learned and target Q-networks respectively. The target network is a copy of the Q-network that is updated less frequently, and using it to compute the target Q-values helps to stabilize the learning process and improve the performance of the DQN algorithm. Note that to increase stability of training we use Huber loss (smooth_l1_loss) instead of L2.

There are several ways to incorporate exploration into the DQN algorithm. One common method is to use an $\epsilon$-greedy exploration strategy, where the agent takes a random action with probability $\epsilon$ and takes the action with the highest estimated Q-value with probability $1 - \epsilon$. The value of $\epsilon$ is typically decreased over time, so that the agent initially explores more and then gradually shifts towards exploitation as it learns more about the environment.


## Double DQN
The loss function of vanilla DQN is defined as the average of single transition temporal difference (TD) error over $B$ transitions:

$$
\mathcal{L}_{\theta} = \frac{1}{B} \sum_{i=1}^{B} \bigl( \mathrm{TD}~(s_i, a_i, s^{'}_{i}) \bigr)^{2}
$$

With transitions $(s_i, a_i, s^{'}_{i})$ sampled uniformly from the experience buffer. The transition TD error is defined through Bellman optimality condition:

$$
\mathrm{TD}~(s_i, a_i, s^{'}_{i}) = Q_{\theta}~(s_i,a_i) - \bigl(r_{(s_i,a_i,s_{i}^{'})} + \gamma ~ \underset{a^{'}_{i} \sim \bar{Q}_{\theta}}{\mathrm{max}} ~ \bar{Q}_{\theta}~(s_{i}^{'},a_{i}^{'}) \bigr)
$$

Where $Q_{\theta}$ and $\bar{Q}_{\theta}$ denote learned and target Q-networks respectively. In the setup above $a_{i}^{'}$ is chosen via maximum operation over the output of the target Q-network for $s^{'}_{i}$. Using a single network to choose the best action and estimate its Q-value promotes overestimated values. Using such values for supervision leads in turn to general overoptimism of the Q-network and is known to sabotage the training.

In Double Deep Q-Network (DDQN) [(van Hasselt 2015)](https://arxiv.org/pdf/1509.06461.pdf) proposes using two Q-networks in the process of target estimation: one Q-network to choose the maximum valued action from (i.e. *argmax*); and the second one to estimate value of the chosen action (i.e. Q-value estimation for the *argmax* result). Authors show that in DDQN estimated Q-values are less likely to be inflated and lead to more stable learning and better policies. We can use $Q_{\theta}$ and $\bar{Q}_{\theta}$ to augment DQN into DDQN:

$$
\mathrm{TD}~(s_i, a_i, s^{'}_{i}) = Q_{\theta}~(s_i,a_i) - \bigl(r_{(s_i,a_i,s_{i}^{'})} + \gamma ~ \bar{Q}_{\theta}~(s_{i}^{'},\underset{a^{'}_{i} \sim Q_{\theta}}{\mathrm{argmax}} ~ Q_{\theta} (s_{i}^{'}, a^{'}_{i})  \bigr)
$$

Such definition of DDQN leads to very small code changes w.r.t. vanilla DQN implementation. Although $Q_{\theta}$ and $\bar{Q}_{\theta}$ are not fully decoupled, using them leads to good performance increase without introduction of additional networks.


## $\mathrm{TD}_{n}$ - N-step Q-value estimation
$N$-step TD ($\mathrm{TD}_{n}$) was introduced long before neural network based RL. In regular TD, we supervise the Q-network with single-step reward summed with highest Q-value of the next state. In contrast to that, $\mathrm{TD}_{n}$ accumulated rewards over $n$ steps and sums it with the highest Q-value of the state that occured after $n$ steps [(Sutton 1988)](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf). Double DQN $\mathrm{TD}_{n}$ loss is defined by:

$$
\mathrm{TD}_{n}(s_i, a_i, s^{'}_{i+n}) = Q_{\theta}~(s_i,a_i) - \biggl(\sum_{k=0}^{n-1} \gamma^{k} ~ r_{(s_{i+k},a_{i+k},s_{i+k}^{'})} + \gamma^{n} \underset{a^{'}_{i+n} \sim \bar{Q}_{\theta}}{\mathrm{max}} ~ \bar{Q}_{\theta}~(s_{i+n}^{'},a_{i+n}^{'}) \biggr)
$$

Implementing $\mathrm{TD}_{n}$ requires changes to the ExperienceBuffer class. We will implement those changes using the **deque** module. This module will store $n$ of the most recent transitions, and will act as an intermediate between agent and buffers main storage. As compared to single step reward and $s_{i}^{'}$ stored by the simple ExperienceBuffer, the main storage of this upgraded buffer should store $n$ step rewards and $s_{i+n}^{'}$.


## Observations
Classic DQN with unoptimized hyperparameters (but reasonable values) the model was able to easily achieve a score oscillating around 200. Since we have a very limited computational budget and the CartPole environment is very popular environment, we used semi-optimized hyperparameters from [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) - with those parameters, the network was able to sometimes achieve score of 500 (max score). However, still on some training run, the network was not able to achieve such great performance. To this end we introduce further methods aimed at reducing variance in performance of the model.

Adding DDQN and TD-N already improves the model such that it almost always reaches max score. Those two improvements are very nice, since they are simple to implement and they have significant impact on the performance of the DQN agent among other classic improvements - this can be observed for example in the RAINBOW paper [(Hessel 2017)](https://arxiv.org/pdf/1710.02298.pdf) - especially TD-N yields significant improvement generally.