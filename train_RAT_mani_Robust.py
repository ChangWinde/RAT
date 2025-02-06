#!/usr/bin/env python3
"""
This is the official implementation RAT-ATLA in the paper (AAAI 2025).

RAT-ATLA is a robust strategy which alternately train an agent and a fixed RAT attacker.

For more details, please refer to our paper: https://arxiv.org/abs/2412.10713

Author: Fengshuo Bai
License: MIT
"""

# Standard library imports
import os
import sys
import time
import math
import copy
import datetime
import pickle as pkl
from collections import deque

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import wandb
import tqdm

# Local imports
from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel
from utils import Controller
from reward_funs import *
import utils

# Dictionary mapping environment names to their MetaWorld counterparts
ENV_NAME_MAPPING = {
    "metaworld_door-lock-v2":    "door-lock-v2",
    "metaworld_door-unlock-v2":  "door-unlock-v2",
    "metaworld_window-open-v2":  "window-open-v2",
    "metaworld_window-close-v2": "window-close-v2",
    "metaworld_drawer-open-v2":  "drawer-open-v2",
    "metaworld_drawer-close-v2": "drawer-close-v2",
    "metaworld_faucet-open-v2":  "faucet-open-v2",
    "metaworld_faucet-close-v2": "faucet-close-v2",
}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class Workspace(object):
    """
    Main workspace class that handles the training process of RAT-ATLA.
    This class coordinates the interaction between the intention model,
    adversarial attacker, and victim model.
    """
    
    def __init__(self, cfg):
        """
        Initialize the workspace with configuration parameters.
        
        Args:
            cfg: Hydra configuration object containing all parameters
        """
        self.work_dir = os.getcwd()
        self.work_dir += '_' + datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')
        os.mkdir(self.work_dir)
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name
        )

        utils.set_seed_everywhere(cfg.seed)
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(cfg.device)
        self.log_success = False
        self.adv_eps = cfg.adv_eps
        self.reward_fn = rf_dict[cfg.env]

        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.attacker_env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)

        # config for victim
        cfg.victim.params.obs_dim = self.env.observation_space.shape[0]
        cfg.victim.params.action_dim = self.env.action_space.shape[0]
        cfg.victim.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.victim = hydra.utils.instantiate(cfg.victim)
        self.victim_replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device
        )
        print("Robust Policy Training!")
        
        # config for intention
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.intention = hydra.utils.instantiate(cfg.agent)

        # config for adversary
        cfg.attacker.params.obs_dim = self.env.observation_space.shape[0]
        cfg.attacker.params.action_dim = self.env.observation_space.shape[0]
        cfg.attacker.params.action_range = [
            float(self.env.observation_space.low.min()),
            float(self.env.observation_space.high.max())
        ]
        self.attacker = hydra.utils.instantiate(cfg.attacker)
        # self.attacker.load(f"{ROOT_DIR}/model/attack_wocar/{cfg.env}/attacker",1000000)

        # no relabel
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device
        )
        
        meta_file = os.path.join(self.work_dir, 'metadata.pkl')
        pkl.dump({'cfg': self.cfg}, open(meta_file, "wb"))

        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0
        self.robust_step = 0

        # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation,
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch,
            large_batch=cfg.large_batch,
            label_margin=cfg.label_margin,
            teacher_beta=cfg.teacher_beta,
            teacher_gamma=cfg.teacher_gamma,
            teacher_eps_mistake=cfg.teacher_eps_mistake,
            teacher_eps_skip=cfg.teacher_eps_skip,
            teacher_eps_equal=cfg.teacher_eps_equal
        )
        
        # controller
        self.controller = Controller(self.env.observation_space.shape[0], 64, 1, 2)
        
        if self.cfg.enable_wandb:
            self.setup_wandb()

    def setup_wandb(self):
        project = 'RAT'
        group = self.cfg.experiment + '_' +self.cfg.env[10:]
        date = datetime.datetime.strftime(datetime.datetime.now(), '%m%d')
        name = date + '_' + str(self.cfg.max_feedback) + '_' + str(self.cfg.reward_batch)
        name += '_' + str(self.cfg.outer_frequency) + '_' + str(self.cfg.alter_update_num)
        name += '_' + str(self.adv_eps) + '_' + str(self.cfg.seed)
        wandb.init(project=project, entity='crl_adv', group=group, name=name, config=self.cfg, save_code=True)
        os.mkdir(f"{self.work_dir}/victim")
        
    
    def bi_level(self, gradient_update=1):
        """
        Implements bi-level optimization for RAT training.
        Updates both attacker and controller using gradient information.
        
        Args:
            gradient_update: Number of gradient updates to perform
        """
        self.attacker.update_actor_old()
        
        # Inner loop - Update attacker
        if self.step % self.cfg.inner_frequency == 0:
            for i in range(self.cfg.alter_update_num):
                obs, _, _, _, _, _ = self.replay_buffer.sample(self.cfg.attacker.params.batch_size)
                attacker_dist = self.attacker.actor(obs)
                a_mu,a_std = attacker_dist.mean,attacker_dist.scale
                perturbations_state = obs+a_mu*self.adv_eps
                victim_dist = self.victim.actor(perturbations_state)
                union_mu = (victim_dist.mean).clamp(*self.victim.action_range)
                pi_nu_alpha = (union_mu,torch.ones_like(victim_dist.scale,device=self.intention.device))
                # actions from pi_alpha*nu
                with torch.no_grad():
                    intention_dist = self.intention.actor(obs)
                    pi_intention = (intention_dist.mean,intention_dist.scale)
                with torch.no_grad():
                    weights = self.controller.forward(obs)

                kl_attacker_loss = utils.calculate_kl_weight(pi_nu_alpha,pi_intention,weights)
                # optimize the actor of intention
                self.attacker.actor_optimizer.zero_grad()
                kl_attacker_loss.backward()
                self.attacker.actor_optimizer.step()
            if self.cfg.enable_wandb:
                wandb.log({'train_actor/kl_attacker_loss': kl_attacker_loss}, step=self.step)

        # Outer loop - Update controller using bi-level gradients 
        if self.step % self.cfg.outer_frequency == 0:
            obs, _, _, _, _, _ = self.replay_buffer.sample(self.cfg.attacker.params.batch_size)
            attacker_dist = self.attacker.actor(obs)
            perturbations_mean = attacker_dist.rsample()
            log_prob = attacker_dist.log_prob(perturbations_mean).sum(-1, keepdim=True)
            perturbations_mean = torch.nn.functional.hardtanh(perturbations_mean)
            perturbations_state = obs+perturbations_mean*self.adv_eps
            victim_dist = self.victim.actor(perturbations_state)
            victim_action = (victim_dist.mean).clamp(*self.victim.action_range)

            actor_Q1, actor_Q2 = self.intention.critic(obs, victim_action)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            outer_loss = (self.intention.alpha.detach() * log_prob - actor_Q).mean()
            out_grad = torch.autograd.grad(outer_loss,self.attacker.actor.parameters())
            
            # inner grad
            attacker_dist = self.attacker.actor_old(obs)
            a_mu,a_std = attacker_dist.mean,attacker_dist.scale
            perturbations_state = obs+a_mu*self.adv_eps
            victim_dist = self.victim.actor(perturbations_state)
            union_mu = (victim_dist.mean).clamp(*self.victim.action_range)
            pi_nu_alpha = (union_mu,torch.ones_like(victim_dist.scale,device=self.intention.device))
            # actions from pi_alpha*nu
            with torch.no_grad():
                intention_dist = self.intention.actor(obs)
                pi_intention = (intention_dist.mean,intention_dist.scale)
            kl_attacker_loss = utils.calculate_kl(pi_nu_alpha,pi_intention)
            kl_grad = torch.autograd.grad(kl_attacker_loss,self.attacker.actor_old.parameters())

            # h(s)
            weight = self.controller.forward(obs).mean()
            product_dot = 0
            for g1,g2 in zip(out_grad,kl_grad):
                product_dot += (g1*g2).sum()
            h = -self.cfg.attacker.params.actor_lr * product_dot
            loss_weight = h*weight

            self.controller.optimizer.zero_grad()
            loss_weight.backward()
            self.controller.optimizer.step()

    def evaluate(self):
        """
        Evaluates the current policy by running episodes without training.
        Logs metrics like rewards and success rates.
        """
        attacker_average_episode_reward = 0
        true_average_episode_reward = 0
        if self.log_success: attack_average_success_rate = 0

        for _ in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            tcp = self.env.tcp_center
            self.attacker.reset()
            done = False
            attacker_episode_reward = 0
            true_episode_reward = 0
            if self.log_success: attack_success_rate = 0

            while not done:
                with utils.eval_mode(self.attacker):
                    perturbations_mean = self.attacker.act(obs, sample=False)
                    perturbations_mean = torch.nn.functional.hardtanh(torch.Tensor(perturbations_mean)).data.numpy()
                    perturbations_state = obs+perturbations_mean*self.adv_eps
                
                victim_action = self.victim.act(perturbations_state,False)
                next_obs,_, done,extra = self.env.step(victim_action)
                next_tcp = self.env.tcp_center
                adv_reward = self.reward_fn(tcp,next_tcp)
                obs = next_obs
                tcp = next_tcp

                attacker_episode_reward += adv_reward
                true_episode_reward += adv_reward
            if self.log_success: 
                attack_success_rate = max(attack_success_rate, 1 if self.reward_fn(None,self.env.tcp_center,True) else 0)

            attacker_average_episode_reward += attacker_episode_reward
            attack_average_success_rate += attack_success_rate
            if self.log_success: true_average_episode_reward += true_episode_reward

        attacker_average_episode_reward /= self.cfg.num_eval_episodes
        true_average_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            attack_average_success_rate /= self.cfg.num_eval_episodes
            attack_average_success_rate *= 100

        self.logger.log('eval/attacker_average_episode_reward', attacker_average_episode_reward, self.step)
        self.logger.log('eval/attack_success_rate', attack_average_success_rate, self.step)
        self.logger.log('eval/true_average_episode_reward', true_average_episode_reward, self.step)
        
        if self.cfg.enable_wandb:
            wandb.log({
                'eval/attacker_average_episode_reward': attacker_average_episode_reward,
                'eval/attack_success_rate': attack_average_success_rate,
                'eval/true_average_episode_reward': true_average_episode_reward
            }, step=self.step)
            
        self.logger.dump(self.step)
    
    def robust_evaluate(self):
        """
        Evaluates the robustness of victim policy against adversarial attacks.
        """
        true_average_episode_reward = 0
        if self.log_success: attack_average_success_rate = 0

        for _ in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.victim.reset()
            done = False
            true_episode_reward = 0
            if self.log_success: success_rate = 0

            while not done:
                with torch.no_grad():
                    perturbations_mean = self.attacker.act(obs, sample=False)
                    perturbations_mean = torch.nn.functional.hardtanh(torch.Tensor(perturbations_mean)).data.numpy()
                    perturbations_state = obs+perturbations_mean*self.adv_eps
                with utils.eval_mode(self.victim):
                    victim_action = self.victim.act(perturbations_state,False)
                
                obs,victim_reward, done,extra = self.env.step(victim_action)
                true_episode_reward += victim_reward
            if self.log_success: 
                success_rate = max(success_rate, extra['success'])

            true_average_episode_reward += true_episode_reward
            attack_average_success_rate += success_rate

        true_average_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            attack_average_success_rate /= self.cfg.num_eval_episodes
            attack_average_success_rate *= 100

        if self.cfg.enable_wandb:
            wandb.log({
                'eval/robust_average_episode_reward': true_average_episode_reward,
                'eval/robust_success_rate': attack_average_success_rate
            }, step=self.robust_step)

    def learn_reward(self, first_flag=0):
        """
        Updates the reward model using collected feedback.
        
        Args:
            first_flag: Whether this is the first time learning rewards
        """
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError

        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)

                if total_acc > 0.97:
                    break

        print("Reward function is updated!! ACC: " + str(total_acc))

    def train_victim(self):
        """
        Trains the victim policy against adversarial attacks.
        """
        print("+++++++++++++++++ Victim Training ++++++++++++++++++")
        episode_success, episode_reward, done = 0, 0, False
        episode_step = 0

        obs = self.env.reset()
        self.victim.reset()

        while self.robust_step < self.cfg.num_train_steps:
            if done or episode_step >= self.env._max_episode_steps:  # Add max steps check
                if self.robust_step > 0 and self.robust_step % self.cfg.eval_frequency == 0:
                    self.robust_evaluate()

                if self.cfg.enable_wandb:
                    wandb.log({
                        'robust/episode_reward': episode_reward,
                        'robust/episode_success': episode_success
                    }, step=self.robust_step)
                
                # Reset environment and episode variables
                obs = self.env.reset()
                self.victim.reset()
                episode_success = 0
                episode_reward = 0
                episode_step = 0
                done = False
                continue

            # sample action for data collection
            if self.robust_step < self.cfg.num_seed_steps:
                perturbations_state = obs
                robust_action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    perturbations_mean = self.attacker.act(obs, sample=False)
                    perturbations_mean = torch.nn.functional.hardtanh(torch.Tensor(perturbations_mean)).data.numpy()
                    perturbations_state = obs+perturbations_mean*self.adv_eps
                robust_action = self.victim.act(perturbations_state, sample=True)

            if self.robust_step > self.cfg.num_seed_steps:
                self.victim.update(self.victim_replay_buffer, self.logger, self.robust_step, print_flag=False)

            next_obs, victim_reward, done, extra = self.env.step(robust_action)
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += victim_reward

            if self.log_success:
                episode_success = max(episode_success, extra['success'])

            if self.robust_step < self.cfg.num_seed_steps:
                next_perturbations_state = next_obs
            else:
                with torch.no_grad():
                    next_perturbations_mean = self.attacker.act(next_obs, sample=False)
                    next_perturbations_mean = torch.nn.functional.hardtanh(torch.Tensor(next_perturbations_mean)).data.numpy()
                    next_perturbations_state = obs+next_perturbations_mean*self.adv_eps
            
            self.victim_replay_buffer.add(perturbations_state,robust_action,victim_reward,next_perturbations_state,done,done_no_max)

            obs = next_obs
            episode_step += 1
            self.robust_step += 1
        
    def run(self):
        """
        Main training loop that coordinates the training of all components.
        """
        episode, intention_episode_reward, done = 0, 0, True
        if self.log_success: episode_success = 0
        true_episode_reward = 0

        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10)
        start_time = time.time()
        fixed_start_time = time.time()

        interact_count = 0
        while self.step < self.cfg.num_train_steps or self.robust_step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    current_time = time.time()
                    self.logger.log('train/duration', current_time - start_time, self.step)
                    self.logger.log('train/total_duration', current_time - fixed_start_time, self.step)
                    if self.cfg.enable_wandb:
                        wandb.log({'train/duration': current_time - start_time}, step=self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/intention_episode_reward', intention_episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                
                if self.cfg.enable_wandb:
                    wandb.log({
                        'train/intention_episode_reward': intention_episode_reward, 
                        'train/true_episode_reward': true_episode_reward,
                        'train/total_feedback': self.total_feedback
                    }, step=self.step)

                if self.log_success:
                    self.logger.log('train/success_rate', episode_success, self.step)
                    self.logger.log('train/true_episode_success', episode_success, self.step)
                    if self.cfg.enable_wandb:
                        wandb.log({'train/true_episode_success': episode_success}, step=self.step)

                self.train_victim()
                obs = self.env.reset()
                tcp = self.env.tcp_center
                self.intention.reset()
                done = False
                switch_horizon = np.random.randint(0,500)
                avg_train_true_return.append(true_episode_reward)
                intention_episode_reward = 0
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                intention_action = self.env.action_space.sample()
            else:
                # pi_nu_alpha => 1:switch_horizon  pi_theta => switch_horizon+1:500
                if self.step % self.env._max_episode_steps < switch_horizon:
                    with utils.eval_mode(self.attacker):
                        perturbations_mean = self.attacker.act(obs, sample=True)
                        perturbations_mean = torch.nn.functional.hardtanh(torch.Tensor(perturbations_mean)).data.numpy()
                        perturbations_state = obs+perturbations_mean*self.adv_eps
                    with torch.no_grad():
                        intention_action = self.victim.act(perturbations_state,False)
                else:
                    with utils.eval_mode(self.intention):
                        intention_action = self.intention.act(obs, sample=True)

            # run training update
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)

                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)

                # first learn reward
                self.learn_reward(first_flag=1)

                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)

                # reset Q due to unsupervised exploration
                self.intention.reset_critic()
                # update agent
                self.intention.update_after_reset(
                    self.replay_buffer, self.logger, self.step,
                    gradient_update=self.cfg.reset_update,
                    policy_update=True,
                    print_flag=False
                )

                # reset interact_count
                interact_count = 0

            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)

                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.env._max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)

                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)

                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0
                        
                self.intention.update(self.replay_buffer, self.logger, self.step, 1, print_flag=False)
                self.bi_level()
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.intention.update_state_ent(self.replay_buffer, self.logger, self.step, gradient_update=1, K=self.cfg.topK, print_flag=False)

            next_obs,_, done,extra = self.env.step(intention_action)
            next_tcp = self.env.tcp_center
            adv_reward = self.reward_fn(tcp,next_tcp)
            
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, intention_action], axis=-1))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            intention_episode_reward += reward_hat
            true_episode_reward += adv_reward

            if self.log_success:
                episode_success = max(episode_success, 1 if self.reward_fn(None,self.env.tcp_center,True) else 0)

            
            # adding data to the reward training data
            self.reward_model.add_data(obs, intention_action, adv_reward, done)
            self.replay_buffer.add(obs,intention_action,reward_hat,next_obs,done,done_no_max)
            
            obs = next_obs
            tcp = next_tcp
            episode_step += 1
            self.step += 1
            interact_count += 1

        self.attacker.save(f"{self.work_dir}/attacker", self.step)
        self.intention.save(f"{self.work_dir}/intention", self.step)
        self.controller.save(f"{self.work_dir}/controller", self.step)
        self.victim.save(f"{self.work_dir}/victim", self.step)
        self.reward_model.save(self.work_dir, self.step)
        if self.cfg.enable_wandb:
            wandb.save(os.path.join(self.work_dir, 'train') + '.csv')
            wandb.save(os.path.join(self.work_dir, 'eval') + '.csv')
        
@hydra.main(config_path='config/train_RAT.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
