# Report


### State and Action Spaces

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector must be a number between -1 and 1.


### Hyperparameters

BUFFER_SIZE = int(1e5) This is a replay buffer. A large Replay Buffer is important for successful learning.DDPG uses a replay buffer to sample experience to update neural network parameters. During each trajectory roll-out, we save all the experience tuples (state, action, reward, next_state) and store them in a finite-sized cache — a “replay buffer.” Then, we sample random mini-batches of experience from the replay buffer when we update the value and policy networks. 

BATCH_SIZE = 256 - the size of the batch for learning the policy. The batch size parameter plays an important role in DDPG. Larger batch size allows more samples to be seen. This [article](https://arxiv.org/pdf/1708.04133.pdf) suggests that larger batch size improves performance of DDPG algoritm.

GAMMA = 0.99 - discount factor. The discount factor is a measure of how far ahead in time the algorithm looks. Nearly always arbitrarily chosen by researchers to be near the 0.9 point. 

TAU = 1e-3  - for soft update of target parameters

LR_ACTOR = 1e-4  and LR_CRITIC = 1e-4 - learning rates of the actor and learning rate of the critic, number that is used for adam optimizer. [Article](https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2)

WEIGHT_DECAY = 0.0 with weight decay set to zero, I got better performance.

N_LEARN_UPDATES = 10  is the number of learning updates

N_TIME_STEPS = 20 means to do an update every n time step



```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay

N_LEARN_UPDATES = 10     # number of learning updates
N_TIME_STEPS = 20       # every n time step do update

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

### Neural networks

The Actor model consists of two fully connected layers with 256 and 128 units with relu activation and tanh activation. The network's initial dimension is the same as the state size. Relu activation and tanh activation funtions are used


The Critic model is similar to Actor model, it has two fully connected layers with 256 and 128 units with leaky_relu activation. The critic network's initial dimension is the sum of the state size and action size. Leaky_relu activation is used


### Learning algorithm

The learning algorithm is based on the paper [“Continuous control with deep reinforcement learning”](https://arxiv.org/abs/1509.02971)

DDPG is a model-free policy based learning algorithm in which the agent learns directly from the un-processed observation spaces through policy gradient method tha estimates the weights of an optimal policy using gradient ascent. Gradient ascent is similar to gradient descent used in neural network.  

DDPG employs Actor-Critic model where Critic model learns the value function to determine how the Actor’s policy based model should change. The Actor brings the advantage of learning in continuous actions space without the need for extra layer of optimization procedures required in a value based function while the Critic supplies the Actor with knowledge of the performance.

Agent continues its training until it solves the environment or reaches maximumum number of episodes, which I set to 5000. The environment is solved when the average reward over the last 100 episodes is at least 30.0. 

Episode keeps running until max_t time-steps is reached or until the environment says it's done.

A reward of +0.1 is provided for each step that the agent's hand is in the goal location.


### Training with ddpg


```python
def ddpg(n_episodes=5000, max_t=1000):
    """ Deep Deterministic Policy Gradients
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    scores_window = deque(maxlen=100)
    scores = np.zeros(num_agents)
    scores_episode = []
    
    agents =[] 
    
    for i in range(num_agents):
        agents.append(Agent(state_size, action_size, random_seed=0))
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        
        for agent in agents:
            agent.reset()
            
        scores = np.zeros(num_agents)
            
        for t in range(max_t):
            #actions = [agents[i].act(states[i]) for i in range(num_agents)]
            actions = np.array([agents[i].act(states[i]) for i in range(num_agents)])
#             if t == 0:
#                 print("actions", actions)
            env_info = env.step(actions)[brain_name]        # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done        
            
            for i in range(num_agents):
                agents[i].step(t,states[i], actions[i], rewards[i], next_states[i], dones[i]) 
 
            states = next_states
            scores += rewards
            if t % 20:
                print('\rTimestep {}\tScore: {:.2f}\tmin: {:.2f}\tmax: {:.2f}'
                      .format(t, np.mean(scores), np.min(scores), np.max(scores)), end="") 
            if np.any(dones):
                break 
        score = np.mean(scores)
        scores_window.append(score)       # save most recent score
        scores_episode.append(score)

        print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, score, np.mean(scores_window)), end="\n")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(Agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(Agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
            
    return scores_episode

scores = ddpg()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
```

    Initialising ReplayBuffer
    Episode 1	Score: 0.00	Average Score: 0.00.00
    Episode 2	Score: 0.00	Average Score: 0.00.00
    Episode 3	Score: 0.00	Average Score: 0.00.00
    Episode 4	Score: 0.05	Average Score: 0.01.05
    Episode 5	Score: 0.42	Average Score: 0.09.42
    Episode 6	Score: 0.00	Average Score: 0.08.00
    Episode 7	Score: 0.55	Average Score: 0.15.55
    Episode 8	Score: 1.33	Average Score: 0.29.33
    Episode 9	Score: 0.00	Average Score: 0.26.00
    Episode 10	Score: 0.70	Average Score: 0.3070
    Episode 11	Score: 0.34	Average Score: 0.3134
    Episode 12	Score: 0.00	Average Score: 0.2800
    Episode 13	Score: 0.07	Average Score: 0.2707
    Episode 14	Score: 0.33	Average Score: 0.2733
    Episode 15	Score: 1.39	Average Score: 0.3539
    Episode 16	Score: 0.14	Average Score: 0.3314
    Episode 17	Score: 0.58	Average Score: 0.3558
    Episode 18	Score: 0.54	Average Score: 0.3654
    Episode 19	Score: 0.11	Average Score: 0.3411
    Episode 20	Score: 0.02	Average Score: 0.3302
    Episode 21	Score: 0.00	Average Score: 0.3100
    Episode 22	Score: 0.00	Average Score: 0.3000
    Episode 23	Score: 0.71	Average Score: 0.3271
    Episode 24	Score: 0.00	Average Score: 0.3000
    Episode 25	Score: 0.42	Average Score: 0.3142
    Episode 26	Score: 0.42	Average Score: 0.3142
    Episode 27	Score: 0.47	Average Score: 0.3247
    Episode 28	Score: 0.73	Average Score: 0.3373
    Episode 29	Score: 1.70	Average Score: 0.3870
    Episode 30	Score: 0.48	Average Score: 0.3848
    Episode 31	Score: 1.52	Average Score: 0.4252
    Episode 32	Score: 0.17	Average Score: 0.4117
    Episode 33	Score: 1.03	Average Score: 0.4303
    Episode 34	Score: 1.01	Average Score: 0.4501
    Episode 35	Score: 0.14	Average Score: 0.4414
    Episode 36	Score: 0.71	Average Score: 0.4571
    Episode 37	Score: 0.41	Average Score: 0.4541
    Episode 38	Score: 0.26	Average Score: 0.4426
    Episode 39	Score: 0.37	Average Score: 0.4437
    Episode 40	Score: 0.97	Average Score: 0.4597
    Episode 41	Score: 1.29	Average Score: 0.4729
    Episode 42	Score: 1.14	Average Score: 0.4914
    Episode 43	Score: 2.62	Average Score: 0.5462
    Episode 44	Score: 1.57	Average Score: 0.5657
    Episode 45	Score: 1.51	Average Score: 0.5851
    Episode 46	Score: 1.00	Average Score: 0.5900
    Episode 47	Score: 0.73	Average Score: 0.5973
    Episode 48	Score: 1.58	Average Score: 0.6258
    Episode 49	Score: 0.20	Average Score: 0.6120
    Episode 50	Score: 0.60	Average Score: 0.6160
    Episode 51	Score: 0.66	Average Score: 0.6166
    Episode 52	Score: 0.51	Average Score: 0.6151
    Episode 53	Score: 1.95	Average Score: 0.6395
    Episode 54	Score: 1.01	Average Score: 0.6401
    Episode 55	Score: 1.71	Average Score: 0.6671
    Episode 56	Score: 2.03	Average Score: 0.6803
    Episode 57	Score: 1.54	Average Score: 0.7054
    Episode 58	Score: 1.30	Average Score: 0.7130
    Episode 59	Score: 0.29	Average Score: 0.7029
    Episode 60	Score: 0.53	Average Score: 0.7053
    Episode 61	Score: 2.47	Average Score: 0.7347
    Episode 62	Score: 1.04	Average Score: 0.7304
    Episode 63	Score: 0.65	Average Score: 0.7365
    Episode 64	Score: 2.23	Average Score: 0.7523
    Episode 65	Score: 1.79	Average Score: 0.7779
    Episode 66	Score: 0.98	Average Score: 0.7798
    Episode 67	Score: 0.97	Average Score: 0.7897
    Episode 68	Score: 3.06	Average Score: 0.8106
    Episode 69	Score: 2.58	Average Score: 0.8458
    Episode 70	Score: 1.62	Average Score: 0.8562
    Episode 71	Score: 1.67	Average Score: 0.8667
    Episode 72	Score: 3.27	Average Score: 0.8927
    Episode 73	Score: 2.64	Average Score: 0.9264
    Episode 74	Score: 3.64	Average Score: 0.9564
    Episode 75	Score: 1.61	Average Score: 0.9661
    Episode 76	Score: 3.31	Average Score: 0.9931
    Episode 77	Score: 1.44	Average Score: 1.0044
    Episode 78	Score: 3.09	Average Score: 1.0209
    Episode 79	Score: 1.49	Average Score: 1.0349
    Episode 80	Score: 2.98	Average Score: 1.0598
    Episode 81	Score: 1.05	Average Score: 1.0505
    Episode 82	Score: 3.34	Average Score: 1.0834
    Episode 83	Score: 2.28	Average Score: 1.1028
    Episode 84	Score: 1.36	Average Score: 1.1036
    Episode 85	Score: 3.46	Average Score: 1.1346
    Episode 86	Score: 5.35	Average Score: 1.1835
    Episode 87	Score: 2.36	Average Score: 1.1936
    Episode 88	Score: 4.27	Average Score: 1.2327
    Episode 89	Score: 2.10	Average Score: 1.2410
    Episode 90	Score: 2.85	Average Score: 1.2585
    Episode 91	Score: 2.70	Average Score: 1.2770
    Episode 92	Score: 2.10	Average Score: 1.2810
    Episode 93	Score: 1.94	Average Score: 1.2994
    Episode 94	Score: 1.16	Average Score: 1.2816
    Episode 95	Score: 3.73	Average Score: 1.3173
    Episode 96	Score: 2.35	Average Score: 1.3235
    Episode 97	Score: 2.25	Average Score: 1.3325
    Episode 98	Score: 0.38	Average Score: 1.3238
    Episode 99	Score: 2.47	Average Score: 1.3347
    Episode 100	Score: 4.10	Average Score: 1.360
    Episode 100	Average Score: 1.36
    Episode 101	Score: 2.55	Average Score: 1.395
    Episode 102	Score: 5.14	Average Score: 1.444
    Episode 103	Score: 3.29	Average Score: 1.479
    Episode 104	Score: 2.97	Average Score: 1.507
    Episode 105	Score: 1.99	Average Score: 1.519
    Episode 106	Score: 2.45	Average Score: 1.545
    Episode 107	Score: 1.31	Average Score: 1.551
    Episode 108	Score: 4.74	Average Score: 1.584
    Episode 109	Score: 2.87	Average Score: 1.617
    Episode 110	Score: 1.43	Average Score: 1.623
    Episode 111	Score: 3.58	Average Score: 1.658
    Episode 112	Score: 1.70	Average Score: 1.670
    Episode 113	Score: 1.42	Average Score: 1.682
    Episode 114	Score: 3.52	Average Score: 1.712
    Episode 115	Score: 2.44	Average Score: 1.724
    Episode 116	Score: 5.61	Average Score: 1.781
    Episode 117	Score: 4.17	Average Score: 1.817
    Episode 118	Score: 1.69	Average Score: 1.829
    Episode 119	Score: 0.88	Average Score: 1.838
    Episode 120	Score: 2.82	Average Score: 1.862
    Episode 121	Score: 4.09	Average Score: 1.909
    Episode 122	Score: 4.80	Average Score: 1.950
    Episode 123	Score: 5.00	Average Score: 1.990
    Episode 124	Score: 2.06	Average Score: 2.016
    Episode 125	Score: 2.79	Average Score: 2.049
    Episode 126	Score: 1.60	Average Score: 2.050
    Episode 127	Score: 2.74	Average Score: 2.074
    Episode 128	Score: 4.73	Average Score: 2.113
    Episode 129	Score: 3.75	Average Score: 2.135
    Episode 130	Score: 6.00	Average Score: 2.190
    Episode 131	Score: 3.07	Average Score: 2.207
    Episode 132	Score: 2.97	Average Score: 2.237
    Episode 133	Score: 3.92	Average Score: 2.262
    Episode 134	Score: 1.54	Average Score: 2.264
    Episode 135	Score: 2.18	Average Score: 2.288
    Episode 136	Score: 4.19	Average Score: 2.329
    Episode 137	Score: 2.46	Average Score: 2.346
    Episode 138	Score: 2.90	Average Score: 2.370
    Episode 139	Score: 3.61	Average Score: 2.401
    Episode 140	Score: 1.92	Average Score: 2.412
    Episode 141	Score: 2.71	Average Score: 2.421
    Episode 142	Score: 3.65	Average Score: 2.455
    Episode 143	Score: 1.59	Average Score: 2.449
    Episode 144	Score: 4.28	Average Score: 2.468
    Episode 145	Score: 4.20	Average Score: 2.490
    Episode 146	Score: 2.13	Average Score: 2.503
    Episode 147	Score: 4.44	Average Score: 2.544
    Episode 148	Score: 4.77	Average Score: 2.577
    Episode 149	Score: 2.46	Average Score: 2.596
    Episode 150	Score: 4.09	Average Score: 2.639
    Episode 151	Score: 7.43	Average Score: 2.703
    Episode 152	Score: 4.34	Average Score: 2.734
    Episode 153	Score: 2.66	Average Score: 2.746
    Episode 154	Score: 3.78	Average Score: 2.778
    Episode 155	Score: 4.20	Average Score: 2.790
    Episode 156	Score: 6.09	Average Score: 2.839
    Episode 157	Score: 5.30	Average Score: 2.870
    Episode 158	Score: 5.64	Average Score: 2.924
    Episode 159	Score: 6.10	Average Score: 2.970
    Episode 160	Score: 2.81	Average Score: 3.001
    Episode 161	Score: 5.32	Average Score: 3.032
    Episode 162	Score: 7.19	Average Score: 3.099
    Episode 163	Score: 4.33	Average Score: 3.123
    Episode 164	Score: 6.75	Average Score: 3.175
    Episode 165	Score: 4.72	Average Score: 3.202
    Episode 166	Score: 4.53	Average Score: 3.233
    Episode 167	Score: 8.63	Average Score: 3.313
    Episode 168	Score: 5.15	Average Score: 3.335
    Episode 169	Score: 5.03	Average Score: 3.363
    Episode 170	Score: 2.96	Average Score: 3.376
    Episode 171	Score: 9.02	Average Score: 3.442
    Episode 172	Score: 5.83	Average Score: 3.473
    Episode 173	Score: 5.47	Average Score: 3.507
    Episode 174	Score: 6.79	Average Score: 3.539
    Episode 175	Score: 5.62	Average Score: 3.572
    Episode 176	Score: 6.68	Average Score: 3.608
    Episode 177	Score: 3.91	Average Score: 3.631
    Episode 178	Score: 4.84	Average Score: 3.644
    Episode 179	Score: 5.97	Average Score: 3.697
    Episode 180	Score: 3.07	Average Score: 3.697
    Episode 181	Score: 7.14	Average Score: 3.754
    Episode 182	Score: 5.90	Average Score: 3.780
    Episode 183	Score: 8.80	Average Score: 3.840
    Episode 184	Score: 7.51	Average Score: 3.901
    Episode 185	Score: 7.29	Average Score: 3.949
    Episode 186	Score: 4.96	Average Score: 3.946
    Episode 187	Score: 6.38	Average Score: 3.988
    Episode 188	Score: 5.13	Average Score: 3.993
    Episode 189	Score: 5.65	Average Score: 4.025
    Episode 190	Score: 4.20	Average Score: 4.040
    Episode 191	Score: 7.42	Average Score: 4.082
    Episode 192	Score: 1.68	Average Score: 4.088
    Episode 193	Score: 7.37	Average Score: 4.137
    Episode 194	Score: 5.79	Average Score: 4.189
    Episode 195	Score: 4.50	Average Score: 4.190
    Episode 196	Score: 4.88	Average Score: 4.218
    Episode 197	Score: 5.60	Average Score: 4.250
    Episode 198	Score: 8.79	Average Score: 4.339
    Episode 199	Score: 7.48	Average Score: 4.388
    Episode 200	Score: 10.80	Average Score: 4.45.80
    Episode 200	Average Score: 4.45
    Episode 201	Score: 11.08	Average Score: 4.53.08
    Episode 202	Score: 6.09	Average Score: 4.549
    Episode 203	Score: 9.58	Average Score: 4.608
    Episode 204	Score: 7.08	Average Score: 4.658
    Episode 205	Score: 6.75	Average Score: 4.695
    Episode 206	Score: 9.44	Average Score: 4.764
    Episode 207	Score: 6.14	Average Score: 4.814
    Episode 208	Score: 6.94	Average Score: 4.834
    Episode 209	Score: 7.27	Average Score: 4.887
    Episode 210	Score: 7.58	Average Score: 4.948
    Episode 211	Score: 10.33	Average Score: 5.01.33
    Episode 212	Score: 7.87	Average Score: 5.077
    Episode 213	Score: 8.56	Average Score: 5.146
    Episode 214	Score: 9.19	Average Score: 5.209
    Episode 215	Score: 6.41	Average Score: 5.241
    Episode 216	Score: 11.05	Average Score: 5.29.05
    Episode 217	Score: 3.47	Average Score: 5.287
    Episode 218	Score: 7.89	Average Score: 5.349
    Episode 219	Score: 8.20	Average Score: 5.420
    Episode 220	Score: 9.52	Average Score: 5.492
    Episode 221	Score: 6.34	Average Score: 5.514
    Episode 222	Score: 11.13	Average Score: 5.57.13
    Episode 223	Score: 9.62	Average Score: 5.622
    Episode 224	Score: 7.87	Average Score: 5.687
    Episode 225	Score: 10.47	Average Score: 5.75.47
    Episode 226	Score: 7.85	Average Score: 5.815
    Episode 227	Score: 6.36	Average Score: 5.856
    Episode 228	Score: 8.06	Average Score: 5.886
    Episode 229	Score: 9.84	Average Score: 5.944
    Episode 230	Score: 8.51	Average Score: 5.971
    Episode 231	Score: 5.69	Average Score: 6.009
    Episode 232	Score: 10.75	Average Score: 6.07.75
    Episode 233	Score: 4.16	Average Score: 6.086
    Episode 234	Score: 12.07	Average Score: 6.18.07
    Episode 235	Score: 13.75	Average Score: 6.30.75
    Episode 236	Score: 8.80	Average Score: 6.340
    Episode 237	Score: 7.07	Average Score: 6.397
    Episode 238	Score: 9.07	Average Score: 6.457
    Episode 239	Score: 8.28	Average Score: 6.508
    Episode 240	Score: 8.59	Average Score: 6.569
    Episode 241	Score: 3.23	Average Score: 6.573
    Episode 242	Score: 7.09	Average Score: 6.609
    Episode 243	Score: 8.87	Average Score: 6.687
    Episode 244	Score: 11.12	Average Score: 6.75.12
    Episode 245	Score: 8.65	Average Score: 6.795
    Episode 246	Score: 9.05	Average Score: 6.865
    Episode 247	Score: 9.90	Average Score: 6.910
    Episode 248	Score: 12.80	Average Score: 6.99.80
    Episode 249	Score: 8.54	Average Score: 7.054
    Episode 250	Score: 11.59	Average Score: 7.13.59
    Episode 251	Score: 6.04	Average Score: 7.124
    Episode 252	Score: 8.56	Average Score: 7.166
    Episode 253	Score: 9.02	Average Score: 7.222
    Episode 254	Score: 11.07	Average Score: 7.29.07
    Episode 255	Score: 5.18	Average Score: 7.308
    Episode 256	Score: 10.34	Average Score: 7.35.34
    Episode 257	Score: 7.65	Average Score: 7.375
    Episode 258	Score: 12.39	Average Score: 7.44.39
    Episode 259	Score: 13.27	Average Score: 7.51.27
    Episode 260	Score: 13.95	Average Score: 7.62.95
    Episode 261	Score: 12.30	Average Score: 7.69.30
    Episode 262	Score: 11.78	Average Score: 7.74.78
    Episode 263	Score: 11.22	Average Score: 7.81.22
    Episode 264	Score: 2.45	Average Score: 7.765
    Episode 265	Score: 10.64	Average Score: 7.82.64
    Episode 266	Score: 10.55	Average Score: 7.88.55
    Episode 267	Score: 9.66	Average Score: 7.896
    Episode 268	Score: 10.04	Average Score: 7.94.04
    Episode 269	Score: 5.05	Average Score: 7.945
    Episode 270	Score: 9.44	Average Score: 8.014
    Episode 271	Score: 8.62	Average Score: 8.002
    Episode 272	Score: 8.67	Average Score: 8.037
    Episode 273	Score: 8.82	Average Score: 8.062
    Episode 274	Score: 6.87	Average Score: 8.067
    Episode 275	Score: 11.13	Average Score: 8.12.13
    Episode 276	Score: 9.03	Average Score: 8.143
    Episode 277	Score: 12.32	Average Score: 8.23.32
    Episode 278	Score: 11.76	Average Score: 8.30.76
    Episode 279	Score: 9.36	Average Score: 8.336
    Episode 280	Score: 7.15	Average Score: 8.375
    Episode 281	Score: 9.07	Average Score: 8.397
    Episode 282	Score: 14.65	Average Score: 8.48.65
    Episode 283	Score: 10.80	Average Score: 8.50.80
    Episode 284	Score: 9.55	Average Score: 8.525
    Episode 285	Score: 9.89	Average Score: 8.549
    Episode 286	Score: 12.78	Average Score: 8.62.78
    Episode 287	Score: 11.87	Average Score: 8.68.87
    Episode 288	Score: 8.78	Average Score: 8.718
    Episode 289	Score: 13.42	Average Score: 8.79.42
    Episode 290	Score: 10.56	Average Score: 8.86.56
    Episode 291	Score: 9.54	Average Score: 8.884
    Episode 292	Score: 13.44	Average Score: 8.99.44
    Episode 293	Score: 13.68	Average Score: 9.06.68
    Episode 294	Score: 18.34	Average Score: 9.18.34
    Episode 295	Score: 16.53	Average Score: 9.30.53
    Episode 296	Score: 12.21	Average Score: 9.38.21
    Episode 297	Score: 8.05	Average Score: 9.405
    Episode 298	Score: 13.23	Average Score: 9.45.23
    Episode 299	Score: 11.78	Average Score: 9.49.78
    Episode 300	Score: 9.95	Average Score: 9.485
    Episode 300	Average Score: 9.48
    Episode 301	Score: 14.64	Average Score: 9.52.64
    Episode 302	Score: 10.98	Average Score: 9.56.98
    Episode 303	Score: 12.18	Average Score: 9.59.18
    Episode 304	Score: 13.71	Average Score: 9.66.71
    Episode 305	Score: 11.08	Average Score: 9.70.08
    Episode 306	Score: 13.06	Average Score: 9.74.06
    Episode 307	Score: 14.98	Average Score: 9.82.98
    Episode 308	Score: 7.93	Average Score: 9.833
    Episode 309	Score: 9.20	Average Score: 9.850
    Episode 310	Score: 10.06	Average Score: 9.88.06
    Episode 311	Score: 10.62	Average Score: 9.88.62
    Episode 312	Score: 7.79	Average Score: 9.889
    Episode 313	Score: 14.47	Average Score: 9.94.47
    Episode 314	Score: 13.18	Average Score: 9.98.18
    Episode 315	Score: 9.57	Average Score: 10.01
    Episode 316	Score: 12.65	Average Score: 10.0365
    Episode 317	Score: 4.56	Average Score: 10.04
    Episode 318	Score: 13.09	Average Score: 10.0909
    Episode 319	Score: 8.47	Average Score: 10.09
    Episode 320	Score: 10.77	Average Score: 10.1177
    Episode 321	Score: 22.15	Average Score: 10.2615
    Episode 322	Score: 13.87	Average Score: 10.2987
    Episode 323	Score: 10.85	Average Score: 10.3085
    Episode 324	Score: 10.55	Average Score: 10.3355
    Episode 325	Score: 10.96	Average Score: 10.3496
    Episode 326	Score: 10.55	Average Score: 10.3655
    Episode 327	Score: 11.04	Average Score: 10.4104
    Episode 328	Score: 10.06	Average Score: 10.4306
    Episode 329	Score: 10.05	Average Score: 10.4305
    Episode 330	Score: 15.17	Average Score: 10.5017
    Episode 331	Score: 15.72	Average Score: 10.6072
    Episode 332	Score: 10.18	Average Score: 10.5918
    Episode 333	Score: 10.96	Average Score: 10.6696
    Episode 334	Score: 9.62	Average Score: 10.64
    Episode 335	Score: 14.42	Average Score: 10.6442
    Episode 336	Score: 8.60	Average Score: 10.64
    Episode 337	Score: 9.92	Average Score: 10.67
    Episode 338	Score: 13.54	Average Score: 10.7154
    Episode 339	Score: 8.56	Average Score: 10.72
    Episode 340	Score: 12.16	Average Score: 10.7516
    Episode 341	Score: 10.39	Average Score: 10.8239
    Episode 342	Score: 12.21	Average Score: 10.8721
    Episode 343	Score: 12.61	Average Score: 10.9161
    Episode 344	Score: 13.57	Average Score: 10.9457
    Episode 345	Score: 9.55	Average Score: 10.95
    Episode 346	Score: 5.35	Average Score: 10.91
    Episode 347	Score: 9.54	Average Score: 10.91
    Episode 348	Score: 10.25	Average Score: 10.8825
    Episode 349	Score: 14.93	Average Score: 10.9493
    Episode 350	Score: 11.17	Average Score: 10.9417
    Episode 351	Score: 15.31	Average Score: 11.0331
    Episode 352	Score: 11.19	Average Score: 11.0619
    Episode 353	Score: 12.15	Average Score: 11.0915
    Episode 354	Score: 7.23	Average Score: 11.05
    Episode 355	Score: 14.93	Average Score: 11.1593
    Episode 356	Score: 11.21	Average Score: 11.1621
    Episode 357	Score: 8.98	Average Score: 11.17
    Episode 358	Score: 6.62	Average Score: 11.11
    Episode 359	Score: 16.35	Average Score: 11.1435
    Episode 360	Score: 9.99	Average Score: 11.10
    Episode 361	Score: 10.24	Average Score: 11.0824
    Episode 362	Score: 6.18	Average Score: 11.03
    Episode 363	Score: 13.11	Average Score: 11.0511
    Episode 364	Score: 13.20	Average Score: 11.1520
    Episode 365	Score: 10.06	Average Score: 11.1506
    Episode 366	Score: 13.03	Average Score: 11.1703
    Episode 367	Score: 8.32	Average Score: 11.16
    Episode 368	Score: 11.67	Average Score: 11.1867
    Episode 369	Score: 10.30	Average Score: 11.2330
    Episode 370	Score: 13.59	Average Score: 11.2759
    Episode 371	Score: 11.99	Average Score: 11.3099
    Episode 372	Score: 11.62	Average Score: 11.3362
    Episode 373	Score: 11.23	Average Score: 11.3623
    Episode 374	Score: 16.47	Average Score: 11.4547
    Episode 375	Score: 10.53	Average Score: 11.4553
    Episode 376	Score: 7.92	Average Score: 11.44
    Episode 377	Score: 10.47	Average Score: 11.4247
    Episode 378	Score: 13.62	Average Score: 11.4462
    Episode 379	Score: 14.28	Average Score: 11.4928
    Episode 380	Score: 12.38	Average Score: 11.5438
    Episode 381	Score: 13.38	Average Score: 11.5838
    Episode 382	Score: 13.63	Average Score: 11.5763
    Episode 383	Score: 16.89	Average Score: 11.6389
    Episode 384	Score: 9.33	Average Score: 11.63
    Episode 385	Score: 9.93	Average Score: 11.63
    Episode 386	Score: 8.90	Average Score: 11.59
    Episode 387	Score: 5.66	Average Score: 11.53
    Episode 388	Score: 7.40	Average Score: 11.52
    Episode 389	Score: 13.84	Average Score: 11.5284
    Episode 390	Score: 7.30	Average Score: 11.49
    Episode 391	Score: 5.90	Average Score: 11.45
    Episode 392	Score: 12.51	Average Score: 11.4451
    Episode 393	Score: 17.77	Average Score: 11.4877
    Episode 394	Score: 9.48	Average Score: 11.39
    Episode 395	Score: 7.81	Average Score: 11.31
    Episode 396	Score: 10.27	Average Score: 11.2927
    Episode 397	Score: 12.01	Average Score: 11.3301
    Episode 398	Score: 13.33	Average Score: 11.3333
    Episode 399	Score: 12.48	Average Score: 11.3348
    Episode 400	Score: 7.91	Average Score: 11.31
    Episode 400	Average Score: 11.31
    Episode 401	Score: 14.31	Average Score: 11.3131
    Episode 402	Score: 19.13	Average Score: 11.3913
    Episode 403	Score: 12.46	Average Score: 11.3946
    Episode 404	Score: 15.90	Average Score: 11.4290
    Episode 405	Score: 17.46	Average Score: 11.4846
    Episode 406	Score: 16.66	Average Score: 11.5266
    Episode 407	Score: 14.80	Average Score: 11.5180
    Episode 408	Score: 13.84	Average Score: 11.5784
    Episode 409	Score: 10.83	Average Score: 11.5983
    Episode 410	Score: 13.88	Average Score: 11.6388
    Episode 411	Score: 12.28	Average Score: 11.6428
    Episode 412	Score: 9.57	Average Score: 11.66
    Episode 413	Score: 16.82	Average Score: 11.6982
    Episode 414	Score: 12.96	Average Score: 11.6896
    Episode 415	Score: 13.39	Average Score: 11.7239
    Episode 416	Score: 14.59	Average Score: 11.7459
    Episode 417	Score: 8.56	Average Score: 11.78
    Episode 418	Score: 22.68	Average Score: 11.8868
    Episode 419	Score: 17.33	Average Score: 11.9733
    Episode 420	Score: 20.20	Average Score: 12.0620
    Episode 421	Score: 15.84	Average Score: 12.0084
    Episode 422	Score: 13.49	Average Score: 11.9949
    Episode 423	Score: 12.58	Average Score: 12.0158
    Episode 424	Score: 12.64	Average Score: 12.0364
    Episode 425	Score: 6.99	Average Score: 11.99
    Episode 426	Score: 7.19	Average Score: 11.96
    Episode 427	Score: 10.56	Average Score: 11.9556
    Episode 428	Score: 8.70	Average Score: 11.94
    Episode 429	Score: 14.55	Average Score: 11.9955
    Episode 430	Score: 11.64	Average Score: 11.9564
    Episode 431	Score: 11.29	Average Score: 11.9129
    Episode 432	Score: 11.25	Average Score: 11.9225
    Episode 433	Score: 11.11	Average Score: 11.9211
    Episode 434	Score: 12.93	Average Score: 11.9593
    Episode 435	Score: 9.91	Average Score: 11.91
    Episode 436	Score: 12.51	Average Score: 11.9451
    Episode 437	Score: 16.30	Average Score: 12.0130
    Episode 438	Score: 18.55	Average Score: 12.0655
    Episode 439	Score: 13.07	Average Score: 12.1007
    Episode 440	Score: 11.55	Average Score: 12.1055
    Episode 441	Score: 9.32	Average Score: 12.09
    Episode 442	Score: 14.28	Average Score: 12.1128
    Episode 443	Score: 15.25	Average Score: 12.1325
    Episode 444	Score: 16.42	Average Score: 12.1642
    Episode 445	Score: 13.78	Average Score: 12.2078
    Episode 446	Score: 18.61	Average Score: 12.3461
    Episode 447	Score: 18.08	Average Score: 12.4208
    Episode 448	Score: 14.14	Average Score: 12.4614
    Episode 449	Score: 10.52	Average Score: 12.4252
    Episode 450	Score: 15.78	Average Score: 12.4678
    Episode 451	Score: 17.42	Average Score: 12.4842
    Episode 452	Score: 15.40	Average Score: 12.5340
    Episode 453	Score: 18.52	Average Score: 12.5952
    Episode 454	Score: 16.82	Average Score: 12.6982
    Episode 455	Score: 13.48	Average Score: 12.6748
    Episode 456	Score: 23.56	Average Score: 12.8056
    Episode 457	Score: 19.21	Average Score: 12.9021
    Episode 458	Score: 15.39	Average Score: 12.9939
    Episode 459	Score: 15.08	Average Score: 12.9708
    Episode 460	Score: 23.19	Average Score: 13.1019
    Episode 461	Score: 29.32	Average Score: 13.3032
    Episode 462	Score: 13.45	Average Score: 13.3745
    Episode 463	Score: 27.67	Average Score: 13.5167
    Episode 464	Score: 13.39	Average Score: 13.5239
    Episode 465	Score: 17.94	Average Score: 13.5994
    Episode 466	Score: 20.13	Average Score: 13.6713
    Episode 467	Score: 23.65	Average Score: 13.8265
    Episode 468	Score: 19.69	Average Score: 13.9069
    Episode 469	Score: 17.45	Average Score: 13.9745
    Episode 470	Score: 22.08	Average Score: 14.0608
    Episode 471	Score: 14.65	Average Score: 14.0865
    Episode 472	Score: 15.18	Average Score: 14.1218
    Episode 473	Score: 21.53	Average Score: 14.2253
    Episode 474	Score: 13.56	Average Score: 14.1956
    Episode 475	Score: 15.73	Average Score: 14.2473
    Episode 476	Score: 12.26	Average Score: 14.2926
    Episode 477	Score: 11.59	Average Score: 14.3059
    Episode 478	Score: 12.09	Average Score: 14.2809
    Episode 479	Score: 16.13	Average Score: 14.3013
    Episode 480	Score: 11.52	Average Score: 14.2952
    Episode 481	Score: 16.41	Average Score: 14.3241
    Episode 482	Score: 14.98	Average Score: 14.3498
    Episode 483	Score: 13.17	Average Score: 14.3017
    Episode 484	Score: 15.97	Average Score: 14.3797
    Episode 485	Score: 19.09	Average Score: 14.4609
    Episode 486	Score: 20.16	Average Score: 14.5716
    Episode 487	Score: 14.50	Average Score: 14.6650
    Episode 488	Score: 14.87	Average Score: 14.7387
    Episode 489	Score: 11.24	Average Score: 14.7124
    Episode 490	Score: 21.40	Average Score: 14.8540
    Episode 491	Score: 10.97	Average Score: 14.9097
    Episode 492	Score: 22.96	Average Score: 15.0096
    Episode 493	Score: 12.33	Average Score: 14.9533
    Episode 494	Score: 21.86	Average Score: 15.0786
    Episode 495	Score: 22.96	Average Score: 15.2296
    Episode 496	Score: 25.16	Average Score: 15.3716
    Episode 497	Score: 18.14	Average Score: 15.4314
    Episode 498	Score: 17.82	Average Score: 15.4882
    Episode 499	Score: 18.17	Average Score: 15.5417
    Episode 500	Score: 9.78	Average Score: 15.55
    Episode 500	Average Score: 15.55
    Episode 501	Score: 20.73	Average Score: 15.6273
    Episode 502	Score: 20.24	Average Score: 15.6324
    Episode 503	Score: 15.26	Average Score: 15.6626
    Episode 504	Score: 14.61	Average Score: 15.6561
    Episode 505	Score: 27.05	Average Score: 15.7405
    Episode 506	Score: 11.20	Average Score: 15.6920
    Episode 507	Score: 23.91	Average Score: 15.7891
    Episode 508	Score: 18.67	Average Score: 15.8367
    Episode 509	Score: 34.66	Average Score: 16.0666
    Episode 510	Score: 17.11	Average Score: 16.1011
    Episode 511	Score: 16.04	Average Score: 16.1304
    Episode 512	Score: 25.55	Average Score: 16.2955
    Episode 513	Score: 15.94	Average Score: 16.2994
    Episode 514	Score: 21.18	Average Score: 16.3718
    Episode 515	Score: 16.06	Average Score: 16.3906
    Episode 516	Score: 28.81	Average Score: 16.5481
    Episode 517	Score: 24.27	Average Score: 16.6927
    Episode 518	Score: 14.80	Average Score: 16.6180
    Episode 519	Score: 27.54	Average Score: 16.7254
    Episode 520	Score: 11.40	Average Score: 16.6340
    Episode 521	Score: 27.94	Average Score: 16.7594
    Episode 522	Score: 23.22	Average Score: 16.8522
    Episode 523	Score: 11.57	Average Score: 16.8457
    Episode 524	Score: 25.40	Average Score: 16.9640
    Episode 525	Score: 20.93	Average Score: 17.1093
    Episode 526	Score: 28.41	Average Score: 17.3241
    Episode 527	Score: 27.64	Average Score: 17.4964
    Episode 528	Score: 22.28	Average Score: 17.6228
    Episode 529	Score: 23.11	Average Score: 17.7111
    Episode 530	Score: 14.46	Average Score: 17.7446
    Episode 531	Score: 25.87	Average Score: 17.8887
    Episode 532	Score: 27.01	Average Score: 18.0401
    Episode 533	Score: 16.58	Average Score: 18.0958
    Episode 534	Score: 19.43	Average Score: 18.1643
    Episode 535	Score: 37.40	Average Score: 18.4340
    Episode 536	Score: 23.55	Average Score: 18.5455
    Episode 537	Score: 22.27	Average Score: 18.6027
    Episode 538	Score: 16.93	Average Score: 18.5993
    Episode 539	Score: 13.16	Average Score: 18.5916
    Episode 540	Score: 25.87	Average Score: 18.7387
    Episode 541	Score: 19.87	Average Score: 18.8487
    Episode 542	Score: 26.52	Average Score: 18.9652
    Episode 543	Score: 26.78	Average Score: 19.0878
    Episode 544	Score: 22.42	Average Score: 19.1442
    Episode 545	Score: 27.64	Average Score: 19.2764
    Episode 546	Score: 15.80	Average Score: 19.2580
    Episode 547	Score: 27.63	Average Score: 19.3463
    Episode 548	Score: 24.59	Average Score: 19.4559
    Episode 549	Score: 30.73	Average Score: 19.6573
    Episode 550	Score: 24.52	Average Score: 19.7452
    Episode 551	Score: 25.17	Average Score: 19.8117
    Episode 552	Score: 21.97	Average Score: 19.8897
    Episode 553	Score: 24.65	Average Score: 19.9465
    Episode 554	Score: 15.35	Average Score: 19.9335
    Episode 555	Score: 29.38	Average Score: 20.0838
    Episode 556	Score: 22.56	Average Score: 20.0756
    Episode 557	Score: 30.33	Average Score: 20.1933
    Episode 558	Score: 20.14	Average Score: 20.2314
    Episode 559	Score: 24.52	Average Score: 20.3352
    Episode 560	Score: 35.32	Average Score: 20.4532
    Episode 561	Score: 28.14	Average Score: 20.4414
    Episode 562	Score: 14.94	Average Score: 20.4594
    Episode 563	Score: 19.56	Average Score: 20.3756
    Episode 564	Score: 15.63	Average Score: 20.3963
    Episode 565	Score: 27.62	Average Score: 20.4962
    Episode 566	Score: 27.18	Average Score: 20.5618
    Episode 567	Score: 24.09	Average Score: 20.5709
    Episode 568	Score: 26.48	Average Score: 20.6348
    Episode 569	Score: 27.34	Average Score: 20.7334
    Episode 570	Score: 19.49	Average Score: 20.7149
    Episode 571	Score: 24.36	Average Score: 20.8036
    Episode 572	Score: 21.29	Average Score: 20.8629
    Episode 573	Score: 19.13	Average Score: 20.8413
    Episode 574	Score: 27.63	Average Score: 20.9863
    Episode 575	Score: 32.72	Average Score: 21.1572
    Episode 576	Score: 23.51	Average Score: 21.2651
    Episode 577	Score: 23.90	Average Score: 21.3990
    Episode 578	Score: 12.47	Average Score: 21.3947
    Episode 579	Score: 23.41	Average Score: 21.4641
    Episode 580	Score: 28.62	Average Score: 21.6362
    Episode 581	Score: 23.30	Average Score: 21.7030
    Episode 582	Score: 21.12	Average Score: 21.7612
    Episode 583	Score: 7.83	Average Score: 21.71
    Episode 584	Score: 30.40	Average Score: 21.8640
    Episode 585	Score: 28.38	Average Score: 21.9538
    Episode 586	Score: 19.55	Average Score: 21.9455
    Episode 587	Score: 26.44	Average Score: 22.0644
    Episode 588	Score: 24.80	Average Score: 22.1680
    Episode 589	Score: 10.53	Average Score: 22.1553
    Episode 590	Score: 26.47	Average Score: 22.2047
    Episode 591	Score: 18.53	Average Score: 22.2853
    Episode 592	Score: 20.61	Average Score: 22.2661
    Episode 593	Score: 18.64	Average Score: 22.3264
    Episode 594	Score: 29.90	Average Score: 22.4090
    Episode 595	Score: 13.67	Average Score: 22.3167
    Episode 596	Score: 17.38	Average Score: 22.2338
    Episode 597	Score: 16.39	Average Score: 22.2139
    Episode 598	Score: 14.49	Average Score: 22.1849
    Episode 599	Score: 19.98	Average Score: 22.2098
    Episode 600	Score: 25.52	Average Score: 22.3552
    Episode 600	Average Score: 22.35
    Episode 601	Score: 20.49	Average Score: 22.3549
    Episode 602	Score: 20.54	Average Score: 22.3554
    Episode 603	Score: 18.44	Average Score: 22.3944
    Episode 604	Score: 23.56	Average Score: 22.4856
    Episode 605	Score: 25.66	Average Score: 22.4666
    Episode 606	Score: 24.53	Average Score: 22.6053
    Episode 607	Score: 25.71	Average Score: 22.6171
    Episode 608	Score: 21.58	Average Score: 22.6458
    Episode 609	Score: 24.47	Average Score: 22.5447
    Episode 610	Score: 25.69	Average Score: 22.6369
    Episode 611	Score: 30.07	Average Score: 22.7707
    Episode 612	Score: 20.11	Average Score: 22.7111
    Episode 613	Score: 25.12	Average Score: 22.8012
    Episode 614	Score: 21.49	Average Score: 22.8149
    Episode 615	Score: 25.72	Average Score: 22.9072
    Episode 616	Score: 21.34	Average Score: 22.8334
    Episode 617	Score: 15.13	Average Score: 22.7413
    Episode 618	Score: 19.21	Average Score: 22.7821
    Episode 619	Score: 29.18	Average Score: 22.8018
    Episode 620	Score: 20.51	Average Score: 22.8951
    Episode 621	Score: 28.70	Average Score: 22.9070
    Episode 622	Score: 23.99	Average Score: 22.9099
    Episode 623	Score: 29.96	Average Score: 23.0996
    Episode 624	Score: 24.70	Average Score: 23.0870
    Episode 625	Score: 26.77	Average Score: 23.1477
    Episode 626	Score: 23.00	Average Score: 23.0900
    Episode 627	Score: 24.79	Average Score: 23.0679
    Episode 628	Score: 27.84	Average Score: 23.1184
    Episode 629	Score: 21.93	Average Score: 23.1093
    Episode 630	Score: 29.57	Average Score: 23.2557
    Episode 631	Score: 27.89	Average Score: 23.2789
    Episode 632	Score: 22.38	Average Score: 23.2338
    Episode 633	Score: 26.10	Average Score: 23.3210
    Episode 634	Score: 13.13	Average Score: 23.2613
    Episode 635	Score: 29.53	Average Score: 23.1853
    Episode 636	Score: 20.07	Average Score: 23.1407
    Episode 637	Score: 24.51	Average Score: 23.1751
    Episode 638	Score: 29.39	Average Score: 23.2939
    Episode 639	Score: 24.36	Average Score: 23.4036
    Episode 640	Score: 30.55	Average Score: 23.4555
    Episode 641	Score: 32.91	Average Score: 23.5891
    Episode 642	Score: 28.82	Average Score: 23.6082
    Episode 643	Score: 28.28	Average Score: 23.6228
    Episode 644	Score: 23.66	Average Score: 23.6366
    Episode 645	Score: 31.60	Average Score: 23.6760
    Episode 646	Score: 35.48	Average Score: 23.8748
    Episode 647	Score: 22.05	Average Score: 23.8105
    Episode 648	Score: 33.01	Average Score: 23.9001
    Episode 649	Score: 31.47	Average Score: 23.9047
    Episode 650	Score: 33.13	Average Score: 23.9913
    Episode 651	Score: 36.18	Average Score: 24.1018
    Episode 652	Score: 38.14	Average Score: 24.2614
    Episode 653	Score: 28.43	Average Score: 24.3043
    Episode 654	Score: 26.34	Average Score: 24.4134
    Episode 655	Score: 33.11	Average Score: 24.4511
    Episode 656	Score: 28.27	Average Score: 24.5027
    Episode 657	Score: 28.20	Average Score: 24.4820
    Episode 658	Score: 33.98	Average Score: 24.6298
    Episode 659	Score: 35.25	Average Score: 24.7325
    Episode 660	Score: 28.83	Average Score: 24.6683
    Episode 661	Score: 31.53	Average Score: 24.7053
    Episode 662	Score: 27.94	Average Score: 24.8394
    Episode 663	Score: 33.24	Average Score: 24.9624
    Episode 664	Score: 32.39	Average Score: 25.1339
    Episode 665	Score: 28.82	Average Score: 25.1482
    Episode 666	Score: 33.52	Average Score: 25.2152
    Episode 667	Score: 29.43	Average Score: 25.2643
    Episode 668	Score: 22.53	Average Score: 25.2253
    Episode 669	Score: 32.26	Average Score: 25.2726
    Episode 670	Score: 26.09	Average Score: 25.3409
    Episode 671	Score: 32.29	Average Score: 25.4129
    Episode 672	Score: 18.52	Average Score: 25.3952
    Episode 673	Score: 33.48	Average Score: 25.5348
    Episode 674	Score: 29.54	Average Score: 25.5554
    Episode 675	Score: 34.66	Average Score: 25.5766
    Episode 676	Score: 28.39	Average Score: 25.6239
    Episode 677	Score: 31.24	Average Score: 25.6924
    Episode 678	Score: 28.06	Average Score: 25.8506
    Episode 679	Score: 28.72	Average Score: 25.9072
    Episode 680	Score: 27.59	Average Score: 25.8959
    Episode 681	Score: 30.89	Average Score: 25.9789
    Episode 682	Score: 29.37	Average Score: 26.0537
    Episode 683	Score: 24.49	Average Score: 26.2249
    Episode 684	Score: 20.21	Average Score: 26.1121
    Episode 685	Score: 29.01	Average Score: 26.1201
    Episode 686	Score: 30.70	Average Score: 26.2370
    Episode 687	Score: 20.02	Average Score: 26.1702
    Episode 688	Score: 14.06	Average Score: 26.0606
    Episode 689	Score: 29.60	Average Score: 26.2560
    Episode 690	Score: 26.77	Average Score: 26.2577
    Episode 691	Score: 26.59	Average Score: 26.3359
    Episode 692	Score: 27.65	Average Score: 26.4065
    Episode 693	Score: 28.07	Average Score: 26.5007
    Episode 694	Score: 29.61	Average Score: 26.5061
    Episode 695	Score: 31.93	Average Score: 26.6893
    Episode 696	Score: 26.90	Average Score: 26.7790
    Episode 697	Score: 29.00	Average Score: 26.9000
    Episode 698	Score: 24.17	Average Score: 27.0017
    Episode 699	Score: 15.46	Average Score: 26.9546
    Episode 700	Score: 25.91	Average Score: 26.9591
    Episode 700	Average Score: 26.95
    Episode 701	Score: 32.28	Average Score: 27.0728
    Episode 702	Score: 27.99	Average Score: 27.1599
    Episode 703	Score: 32.81	Average Score: 27.2981
    Episode 704	Score: 25.43	Average Score: 27.3143
    Episode 705	Score: 27.05	Average Score: 27.3205
    Episode 706	Score: 24.21	Average Score: 27.3221
    Episode 707	Score: 33.71	Average Score: 27.4071
    Episode 708	Score: 31.63	Average Score: 27.5063
    Episode 709	Score: 22.68	Average Score: 27.4868
    Episode 710	Score: 23.32	Average Score: 27.4632
    Episode 711	Score: 28.54	Average Score: 27.4454
    Episode 712	Score: 27.92	Average Score: 27.5292
    Episode 713	Score: 28.79	Average Score: 27.5679
    Episode 714	Score: 26.82	Average Score: 27.6182
    Episode 715	Score: 18.80	Average Score: 27.5480
    Episode 716	Score: 12.77	Average Score: 27.4677
    Episode 717	Score: 22.76	Average Score: 27.5376
    Episode 718	Score: 34.24	Average Score: 27.6824
    Episode 719	Score: 32.09	Average Score: 27.7109
    Episode 720	Score: 33.53	Average Score: 27.8453
    Episode 721	Score: 17.96	Average Score: 27.7496
    Episode 722	Score: 30.15	Average Score: 27.8015
    Episode 723	Score: 17.71	Average Score: 27.6771
    Episode 724	Score: 31.44	Average Score: 27.7444
    Episode 725	Score: 23.40	Average Score: 27.7140
    Episode 726	Score: 26.28	Average Score: 27.7428
    Episode 727	Score: 31.99	Average Score: 27.8199
    Episode 728	Score: 27.15	Average Score: 27.8115
    Episode 729	Score: 28.38	Average Score: 27.8738
    Episode 730	Score: 35.85	Average Score: 27.9385
    Episode 731	Score: 24.40	Average Score: 27.9040
    Episode 732	Score: 27.54	Average Score: 27.9554
    Episode 733	Score: 30.28	Average Score: 27.9928
    Episode 734	Score: 29.17	Average Score: 28.1517
    Episode 735	Score: 31.35	Average Score: 28.1735
    Episode 736	Score: 33.55	Average Score: 28.3155
    Episode 737	Score: 23.35	Average Score: 28.2935
    Episode 738	Score: 31.09	Average Score: 28.3109
    Episode 739	Score: 21.67	Average Score: 28.2867
    Episode 740	Score: 26.19	Average Score: 28.2419
    Episode 741	Score: 17.80	Average Score: 28.0980
    Episode 742	Score: 20.74	Average Score: 28.0174
    Episode 743	Score: 27.73	Average Score: 28.0073
    Episode 744	Score: 26.42	Average Score: 28.0342
    Episode 745	Score: 29.95	Average Score: 28.0195
    Episode 746	Score: 25.45	Average Score: 27.9145
    Episode 747	Score: 34.69	Average Score: 28.0469
    Episode 748	Score: 28.88	Average Score: 28.0088
    Episode 749	Score: 30.16	Average Score: 27.9916
    Episode 750	Score: 25.02	Average Score: 27.9002
    Episode 751	Score: 22.47	Average Score: 27.7747
    Episode 752	Score: 34.43	Average Score: 27.7343
    Episode 753	Score: 26.69	Average Score: 27.7169
    Episode 754	Score: 26.46	Average Score: 27.7146
    Episode 755	Score: 21.63	Average Score: 27.6063
    Episode 756	Score: 18.99	Average Score: 27.5199
    Episode 757	Score: 36.18	Average Score: 27.5918
    Episode 758	Score: 23.32	Average Score: 27.4832
    Episode 759	Score: 22.48	Average Score: 27.3548
    Episode 760	Score: 27.07	Average Score: 27.3307
    Episode 761	Score: 29.72	Average Score: 27.3272
    Episode 762	Score: 30.56	Average Score: 27.3456
    Episode 763	Score: 34.79	Average Score: 27.3679
    Episode 764	Score: 17.14	Average Score: 27.2114
    Episode 765	Score: 32.03	Average Score: 27.2403
    Episode 766	Score: 29.30	Average Score: 27.2030
    Episode 767	Score: 27.83	Average Score: 27.1883
    Episode 768	Score: 27.16	Average Score: 27.2316
    Episode 769	Score: 19.54	Average Score: 27.1054
    Episode 770	Score: 18.96	Average Score: 27.0396
    Episode 771	Score: 29.33	Average Score: 27.0033
    Episode 772	Score: 18.92	Average Score: 27.0092
    Episode 773	Score: 28.62	Average Score: 26.9562
    Episode 774	Score: 27.18	Average Score: 26.9318
    Episode 775	Score: 33.92	Average Score: 26.9292
    Episode 776	Score: 30.39	Average Score: 26.9439
    Episode 777	Score: 25.34	Average Score: 26.8834
    Episode 778	Score: 29.39	Average Score: 26.9039
    Episode 779	Score: 25.57	Average Score: 26.8757
    Episode 780	Score: 31.16	Average Score: 26.9016
    Episode 781	Score: 30.02	Average Score: 26.8902
    Episode 782	Score: 20.92	Average Score: 26.8192
    Episode 783	Score: 29.25	Average Score: 26.8625
    Episode 784	Score: 25.31	Average Score: 26.9131
    Episode 785	Score: 34.92	Average Score: 26.9792
    Episode 786	Score: 35.35	Average Score: 27.0135
    Episode 787	Score: 25.61	Average Score: 27.0761
    Episode 788	Score: 29.91	Average Score: 27.2391
    Episode 789	Score: 19.48	Average Score: 27.1348
    Episode 790	Score: 21.70	Average Score: 27.0770
    Episode 791	Score: 30.22	Average Score: 27.1122
    Episode 792	Score: 29.22	Average Score: 27.1322
    Episode 793	Score: 27.46	Average Score: 27.1246
    Episode 794	Score: 22.05	Average Score: 27.0405
    Episode 795	Score: 26.22	Average Score: 26.9922
    Episode 796	Score: 32.37	Average Score: 27.0437
    Episode 797	Score: 24.82	Average Score: 27.0082
    Episode 798	Score: 25.45	Average Score: 27.0145
    Episode 799	Score: 23.42	Average Score: 27.0942
    Episode 800	Score: 29.42	Average Score: 27.1342
    Episode 800	Average Score: 27.13
    Episode 801	Score: 35.80	Average Score: 27.1680
    Episode 802	Score: 31.95	Average Score: 27.2095
    Episode 803	Score: 23.73	Average Score: 27.1173
    Episode 804	Score: 28.17	Average Score: 27.1417
    Episode 805	Score: 32.50	Average Score: 27.1950
    Episode 806	Score: 33.32	Average Score: 27.2832
    Episode 807	Score: 24.83	Average Score: 27.2083
    Episode 808	Score: 39.57	Average Score: 27.2857
    Episode 809	Score: 34.14	Average Score: 27.3914
    Episode 810	Score: 32.43	Average Score: 27.4843
    Episode 811	Score: 28.57	Average Score: 27.4857
    Episode 812	Score: 27.23	Average Score: 27.4723
    Episode 813	Score: 24.08	Average Score: 27.4308
    Episode 814	Score: 33.78	Average Score: 27.5078
    Episode 815	Score: 28.55	Average Score: 27.5955
    Episode 816	Score: 28.06	Average Score: 27.7506
    Episode 817	Score: 27.50	Average Score: 27.7950
    Episode 818	Score: 26.77	Average Score: 27.7277
    Episode 819	Score: 19.17	Average Score: 27.5917
    Episode 820	Score: 36.02	Average Score: 27.6202
    Episode 821	Score: 26.90	Average Score: 27.7190
    Episode 822	Score: 27.57	Average Score: 27.6857
    Episode 823	Score: 29.66	Average Score: 27.8066
    Episode 824	Score: 27.51	Average Score: 27.7651
    Episode 825	Score: 22.11	Average Score: 27.7511
    Episode 826	Score: 27.38	Average Score: 27.7638
    Episode 827	Score: 22.47	Average Score: 27.6647
    Episode 828	Score: 28.12	Average Score: 27.6712
    Episode 829	Score: 27.15	Average Score: 27.6615
    Episode 830	Score: 22.51	Average Score: 27.5351
    Episode 831	Score: 20.67	Average Score: 27.4967
    Episode 832	Score: 31.70	Average Score: 27.5370
    Episode 833	Score: 32.82	Average Score: 27.5682
    Episode 834	Score: 26.86	Average Score: 27.5386
    Episode 835	Score: 29.93	Average Score: 27.5293
    Episode 836	Score: 25.93	Average Score: 27.4493
    Episode 837	Score: 29.08	Average Score: 27.5008
    Episode 838	Score: 25.16	Average Score: 27.4416
    Episode 839	Score: 31.79	Average Score: 27.5479
    Episode 840	Score: 35.51	Average Score: 27.6451
    Episode 841	Score: 29.04	Average Score: 27.7504
    Episode 842	Score: 30.78	Average Score: 27.8578
    Episode 843	Score: 30.28	Average Score: 27.8728
    Episode 844	Score: 31.48	Average Score: 27.9248
    Episode 845	Score: 23.62	Average Score: 27.8662
    Episode 846	Score: 36.05	Average Score: 27.9705
    Episode 847	Score: 30.26	Average Score: 27.9226
    Episode 848	Score: 30.46	Average Score: 27.9446
    Episode 849	Score: 26.39	Average Score: 27.9039
    Episode 850	Score: 31.51	Average Score: 27.9751
    Episode 851	Score: 21.21	Average Score: 27.9521
    Episode 852	Score: 18.57	Average Score: 27.7957
    Episode 853	Score: 33.31	Average Score: 27.8631
    Episode 854	Score: 30.45	Average Score: 27.9045
    Episode 855	Score: 28.14	Average Score: 27.9714
    Episode 856	Score: 34.40	Average Score: 28.1240
    Episode 857	Score: 31.28	Average Score: 28.0728
    Episode 858	Score: 25.62	Average Score: 28.0962
    Episode 859	Score: 26.00	Average Score: 28.1300
    Episode 860	Score: 29.38	Average Score: 28.1538
    Episode 861	Score: 22.34	Average Score: 28.0834
    Episode 862	Score: 33.67	Average Score: 28.1167
    Episode 863	Score: 34.59	Average Score: 28.1159
    Episode 864	Score: 33.07	Average Score: 28.2707
    Episode 865	Score: 34.80	Average Score: 28.2980
    Episode 866	Score: 30.01	Average Score: 28.3001
    Episode 867	Score: 35.23	Average Score: 28.3823
    Episode 868	Score: 30.17	Average Score: 28.4117
    Episode 869	Score: 35.68	Average Score: 28.5768
    Episode 870	Score: 22.72	Average Score: 28.6072
    Episode 871	Score: 30.03	Average Score: 28.6103
    Episode 872	Score: 36.38	Average Score: 28.7938
    Episode 873	Score: 31.07	Average Score: 28.8107
    Episode 874	Score: 27.97	Average Score: 28.8297
    Episode 875	Score: 31.50	Average Score: 28.7950
    Episode 876	Score: 29.41	Average Score: 28.7841
    Episode 877	Score: 20.32	Average Score: 28.7332
    Episode 878	Score: 35.29	Average Score: 28.7929
    Episode 879	Score: 30.54	Average Score: 28.8454
    Episode 880	Score: 26.88	Average Score: 28.8088
    Episode 881	Score: 35.66	Average Score: 28.8666
    Episode 882	Score: 32.49	Average Score: 28.9749
    Episode 883	Score: 26.21	Average Score: 28.9421
    Episode 884	Score: 27.43	Average Score: 28.9643
    Episode 885	Score: 28.21	Average Score: 28.9021
    Episode 886	Score: 30.77	Average Score: 28.8577
    Episode 887	Score: 34.70	Average Score: 28.9470
    Episode 888	Score: 29.68	Average Score: 28.9468
    Episode 889	Score: 30.81	Average Score: 29.0581
    Episode 890	Score: 32.91	Average Score: 29.1691
    Episode 891	Score: 19.05	Average Score: 29.0505
    Episode 892	Score: 31.67	Average Score: 29.0867
    Episode 893	Score: 30.32	Average Score: 29.1132
    Episode 894	Score: 24.39	Average Score: 29.1339
    Episode 895	Score: 22.57	Average Score: 29.0957
    Episode 896	Score: 31.17	Average Score: 29.0817
    Episode 897	Score: 30.36	Average Score: 29.1436
    Episode 898	Score: 27.80	Average Score: 29.1680
    Episode 899	Score: 28.90	Average Score: 29.2190
    Episode 900	Score: 27.17	Average Score: 29.1917
    Episode 900	Average Score: 29.19
    Episode 901	Score: 18.55	Average Score: 29.0255
    Episode 902	Score: 31.95	Average Score: 29.0295
    Episode 903	Score: 15.66	Average Score: 28.9466
    Episode 904	Score: 30.95	Average Score: 28.9795
    Episode 905	Score: 23.28	Average Score: 28.8728
    Episode 906	Score: 24.12	Average Score: 28.7812
    Episode 907	Score: 16.65	Average Score: 28.7065
    Episode 908	Score: 32.22	Average Score: 28.6322
    Episode 909	Score: 31.50	Average Score: 28.6050
    Episode 910	Score: 24.51	Average Score: 28.5251
    Episode 911	Score: 27.41	Average Score: 28.5141
    Episode 912	Score: 17.11	Average Score: 28.4111
    Episode 913	Score: 23.90	Average Score: 28.4190
    Episode 914	Score: 32.74	Average Score: 28.4074
    Episode 915	Score: 30.66	Average Score: 28.4266
    Episode 916	Score: 21.38	Average Score: 28.3538
    Episode 917	Score: 31.14	Average Score: 28.3914
    Episode 918	Score: 32.30	Average Score: 28.4430
    Episode 919	Score: 35.76	Average Score: 28.6176
    Episode 920	Score: 29.16	Average Score: 28.5416
    Episode 921	Score: 28.23	Average Score: 28.5523
    Episode 922	Score: 29.12	Average Score: 28.5712
    Episode 923	Score: 37.35	Average Score: 28.6535
    Episode 924	Score: 33.77	Average Score: 28.7177
    Episode 925	Score: 33.66	Average Score: 28.8266
    Episode 926	Score: 22.98	Average Score: 28.7898
    Episode 927	Score: 27.18	Average Score: 28.8318
    Episode 928	Score: 24.66	Average Score: 28.7966
    Episode 929	Score: 31.03	Average Score: 28.8303
    Episode 930	Score: 25.61	Average Score: 28.8661
    Episode 931	Score: 31.22	Average Score: 28.9722
    Episode 932	Score: 19.62	Average Score: 28.8562
    Episode 933	Score: 32.92	Average Score: 28.8592
    Episode 934	Score: 15.91	Average Score: 28.7491
    Episode 935	Score: 28.95	Average Score: 28.7395
    Episode 936	Score: 21.43	Average Score: 28.6843
    Episode 937	Score: 36.93	Average Score: 28.7693
    Episode 938	Score: 31.08	Average Score: 28.8208
    Episode 939	Score: 30.81	Average Score: 28.8181
    Episode 940	Score: 30.59	Average Score: 28.7659
    Episode 941	Score: 27.19	Average Score: 28.7419
    Episode 942	Score: 32.84	Average Score: 28.7684
    Episode 943	Score: 29.56	Average Score: 28.7656
    Episode 944	Score: 28.34	Average Score: 28.7334
    Episode 945	Score: 27.09	Average Score: 28.7609
    Episode 946	Score: 28.67	Average Score: 28.6967
    Episode 947	Score: 33.97	Average Score: 28.7297
    Episode 948	Score: 31.21	Average Score: 28.7321
    Episode 949	Score: 24.47	Average Score: 28.7147
    Episode 950	Score: 31.70	Average Score: 28.7170
    Episode 951	Score: 23.00	Average Score: 28.7300
    Episode 952	Score: 33.43	Average Score: 28.8843
    Episode 953	Score: 34.83	Average Score: 28.9083
    Episode 954	Score: 29.96	Average Score: 28.8996
    Episode 955	Score: 34.72	Average Score: 28.9672
    Episode 956	Score: 31.33	Average Score: 28.9333
    Episode 957	Score: 28.46	Average Score: 28.9046
    Episode 958	Score: 24.93	Average Score: 28.8993
    Episode 959	Score: 29.25	Average Score: 28.9225
    Episode 960	Score: 22.52	Average Score: 28.8552
    Episode 961	Score: 29.06	Average Score: 28.9206
    Episode 962	Score: 33.66	Average Score: 28.9266
    Episode 963	Score: 34.69	Average Score: 28.9269
    Episode 964	Score: 23.98	Average Score: 28.8398
    Episode 965	Score: 33.99	Average Score: 28.8299
    Episode 966	Score: 37.11	Average Score: 28.8911
    Episode 967	Score: 36.33	Average Score: 28.9133
    Episode 968	Score: 35.45	Average Score: 28.9645
    Episode 969	Score: 37.30	Average Score: 28.9730
    Episode 970	Score: 27.25	Average Score: 29.0225
    Episode 971	Score: 35.85	Average Score: 29.0885
    Episode 972	Score: 28.01	Average Score: 28.9901
    Episode 973	Score: 36.92	Average Score: 29.0592
    Episode 974	Score: 34.71	Average Score: 29.1271
    Episode 975	Score: 34.26	Average Score: 29.1526
    Episode 976	Score: 32.52	Average Score: 29.1852
    Episode 977	Score: 33.62	Average Score: 29.3162
    Episode 978	Score: 21.62	Average Score: 29.1762
    Episode 979	Score: 25.68	Average Score: 29.1368
    Episode 980	Score: 31.96	Average Score: 29.1896
    Episode 981	Score: 28.43	Average Score: 29.1043
    Episode 982	Score: 32.03	Average Score: 29.1003
    Episode 983	Score: 35.33	Average Score: 29.1933
    Episode 984	Score: 31.43	Average Score: 29.2343
    Episode 985	Score: 18.77	Average Score: 29.1477
    Episode 986	Score: 33.79	Average Score: 29.1779
    Episode 987	Score: 33.52	Average Score: 29.1652
    Episode 988	Score: 34.45	Average Score: 29.2045
    Episode 989	Score: 33.43	Average Score: 29.2343
    Episode 990	Score: 24.64	Average Score: 29.1564
    Episode 991	Score: 26.87	Average Score: 29.2287
    Episode 992	Score: 28.92	Average Score: 29.2092
    Episode 993	Score: 32.11	Average Score: 29.2211
    Episode 994	Score: 28.18	Average Score: 29.2518
    Episode 995	Score: 32.80	Average Score: 29.3680
    Episode 996	Score: 23.00	Average Score: 29.2700
    Episode 997	Score: 26.97	Average Score: 29.2497
    Episode 998	Score: 33.73	Average Score: 29.3073
    Episode 999	Score: 37.81	Average Score: 29.3981
    Episode 1000	Score: 35.26	Average Score: 29.476
    Episode 1000	Average Score: 29.47
    Episode 1001	Score: 29.68	Average Score: 29.588
    Episode 1002	Score: 29.87	Average Score: 29.567
    Episode 1003	Score: 33.24	Average Score: 29.744
    Episode 1004	Score: 26.69	Average Score: 29.699
    Episode 1005	Score: 22.23	Average Score: 29.683
    Episode 1006	Score: 26.47	Average Score: 29.717
    Episode 1007	Score: 25.22	Average Score: 29.792
    Episode 1008	Score: 27.97	Average Score: 29.757
    Episode 1009	Score: 29.18	Average Score: 29.738
    Episode 1010	Score: 31.37	Average Score: 29.797
    Episode 1011	Score: 30.20	Average Score: 29.820
    Episode 1012	Score: 36.30	Average Score: 30.010
    
    Environment solved in 912 episodes!	Average Score: 30.01



![png](output_11_1.png)


### Ideas for improving the agent's performance.

1. Increasing timestep might speedup convergence

2. Changing activation function. This [article](https://arxiv.org/pdf/1804.00361.pdf) suggests that SELU activation function outperformes
Leaky ReLU, Tanh and Sigmoid.

3. Exploring TD3 for implementation [article](https://arxiv.org/pdf/1802.09477.pdf) to adress the unstability and heavy reliance on finding the correct hyper parameters for the current task that DDPG have 




```python

```
