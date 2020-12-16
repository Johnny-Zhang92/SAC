import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
import expert_data
from common import logger
from tqdm import tqdm
import torch.nn as nn
from torch.optim import adadelta
import itertools
import time
import dynamic_model as dm


# Initialize neural network weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# create neural network of dynamic
class ActorNetwork(nn.Module):
    def __init__(self, num_state, num_action, dynamic_hidden_dim):
        super(ActorNetwork, self).__init__()

        self.linear1 = nn.Linear(num_state, dynamic_hidden_dim)
        self.linear2 = nn.Linear(dynamic_hidden_dim, dynamic_hidden_dim)

        self.linear3 = nn.Linear(dynamic_hidden_dim, num_action)

        self.apply(weights_init_)

    # use dynamic neural network to compute next state s'
    def forward(self, state):
        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1

    def sample(self, state):
        mean = self.forward(state)
        # noise = self.noise.normal_(0., std=0.1)
        # noise = noise.clamp(-0.25, 0.25)
        # action = mean + noise
        action = mean
        return action, torch.tensor(0.), mean

    def to(self, device):
        return super(ActorNetwork, self).to(device)


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda:0" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
                self.device)
            self.policy_optim1 = Adam(self.policy.parameters(), lr=args.lr1)
            self.policy_optim = adadelta.Adadelta(self.policy.parameters(),
                                                  lr=args.actor_lr, eps=args.actor_eps)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            # self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(
            #     self.device)
            self.policy = ActorNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
            self.policy_optim1 = Adam(self.policy.parameters(), lr=args.lr1)
            self.policy_optim = adadelta.Adadelta(self.policy.parameters(),
                                                  lr=args.actor_lr, eps=args.actor_eps)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates, bc_agent,
                          model_state_batch=None, model_action_batch=None,
                          real_reward_batch=None, model_next_state_batch=None, model_reward_batch=None,
                          identify=None):
        if not identify:  # sample real next state from memory batch
            # Sample a batch from memory
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, dy_reward_batch \
                = memory.sample(batch_size=batch_size)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            state_batch = torch.FloatTensor(state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            # reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            # print("dy_reward_batch0:", dy_reward_batch)
            dy_reward_batch = torch.FloatTensor(dy_reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

            # print("dy_reward_batch1:", dy_reward_batch.size())
            # print('mask_batch1:', mask_batch.size())
            # # print("reward_batch:", reward_batch.size())
            # print("next_state_batch:", next_state_batch.size())
        else:  # use model to predict next state
            next_state_batch = model_next_state_batch.to(self.device)
            state_batch = model_state_batch.to(self.device)
            action_batch = model_action_batch.to(self.device)
            # reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            # print("model_reward_batch:", model_reward_batch)
            dy_reward_batch = torch.FloatTensor(model_reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.ones(batch_size).to(self.device).unsqueeze(1)
            # print("dy_reward_batch:", dy_reward_batch.size())
            # print("mask_batch:", mask_batch.size())
            # print("next_state_batch1:", next_state_batch.size())
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = dy_reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        with torch.no_grad():
            bc_action_batch = bc_agent.policy(state_batch)
            # print("bc_action_batch, action_batch:", bc_action_batch, '\n', action_batch)
        # select_action_batch = self.policy(state_batch)
        bc_loss_function = nn.MSELoss()
        bc_loss = bc_loss_function(bc_action_batch, pi)
        policy_loss = 0.2*((self.alpha * log_pi) - min_qf_pi).mean() + bc_loss  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # update pretrain actor network parameter
    def update_pretrain_actor_parameters(self, state_batch, action_batch, split='train'):
        select_action_batch = self.policy(state_batch.to(self.device))
        # loss function
        loss_function = nn.MSELoss()
        loss = loss_function(select_action_batch, action_batch.to(self.device))
        if split == "val":
            return loss.item()
        self.policy_optim.zero_grad()  # clean gradient
        loss.backward()  # compute gradient
        self.policy_optim.step()  # update weight
        if split == "train":
            return loss.item()

    # pretrain actor network
    def pre_train_actor(self, expert_path, traj_limitation,
                        pre_train_batchsize=128, max_iters=1e5,
                        verbose=False):
        val_per_iter = int(max_iters / 10)
        dataset = expert_data.MuJoCoExpertData(expert_path=expert_path,
                                               traj_limitation=traj_limitation)
        # start training
        logger.log("pretraining Actor Network with Expert Demonstration")
        for iter_so_far in tqdm(range(int(max_iters))):
            # get train bath_set from expert demonstration(.npz)
            state_batch, action_batch, next_state_batch = \
                dataset.get_next_expert_batch(pre_train_batchsize, 'train')
            # update dynamic network parameters
            train_loss = self.update_pretrain_actor_parameters(state_batch, action_batch, split='train')
            # verify dynamic network performance
            if verbose and iter_so_far % val_per_iter == 0:
                state_batch, action_batch, next_state_batch = \
                    dataset.get_next_expert_batch(-1, 'val')
                val_loss = self.update_pretrain_actor_parameters(state_batch, action_batch, split='val')
                logger.log("Training loss:{},Validation loss:{}".format(train_loss, val_loss))

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))


def test():
    from passer import get_passer
    import gym
    # get args
    args = get_passer()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    # agent.pre_train_actor(expert_path=args.expert_path, traj_limitation=args.traj_limitation,
    #                       pre_train_batchsize=128, max_iters=1e5,
    #                       verbose=True)
    # save actor network
    actor_network_path = "models/{}/actor_{}".format(args.env_name, args.env_name)
    # torch.save(agent.policy.state_dict(), actor_network_path)
    # print('Saving dy_models to {}'.format(actor_network_path))
    # load actor network
    agent.policy.load_state_dict(torch.load(actor_network_path))

    # Training Loop
    total_numsteps = 0
    true_updates = 0
    model_updates = 0
    for i_episode in itertools.count(1):
        episode_reward = 0
        model_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        real_model_loss = None
        while not done:
            action = agent.select_action(state)  # Sample action from policy
            next_state, reward, done, _ = env.step(action)  # Step
            env.render()  # display picture
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            state = next_state
            if total_numsteps > args.num_steps:
                break
            if i_episode % 10 == 0:
                print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, model_reward:{}, "
                      "real_model_loss:{} ".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2),
                       model_reward, real_model_loss))

            if i_episode % 10 == 0 and args.eval is True:
                # print('dynamic_reward.max:', dynamic_reward.mean())
                avg_reward = 0.
                episodes = 10
                for _ in range(episodes):
                    state = env.reset()
                    episode_reward = 0
                    done = False
                    while not done:
                        action = agent.select_action(state, evaluate=True)

                        next_state, reward, done, _ = env.step(action)
                        episode_reward += reward

                        state = next_state
                    avg_reward += episode_reward
                avg_reward /= episodes

                print("----------------------------------------")
                print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
                print("----------------------------------------")

    env.close()


if __name__ == '__main__':
    test()
