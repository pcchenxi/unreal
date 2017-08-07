# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import time
import sys

from environment.environment import Environment
from model.unreal_model import UnrealModel
from trainer.experience import Experience, ExperienceFrame

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000


class Trainer(object):
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               env_type,
               env_name,
               use_pixel_change,
               use_value_replay,
               use_reward_prediction,
               pixel_change_lambda,
               entropy_beta,
               local_t_max,
               gamma,
               gamma_pc,
               experience_history_size,
               max_global_time_step,
               device):

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.env_type = env_type
    self.env_name = env_name
    self.use_pixel_change = use_pixel_change
    self.use_value_replay = use_value_replay
    self.use_reward_prediction = use_reward_prediction
    self.local_t_max = local_t_max
    self.gamma = gamma
    self.gamma_pc = gamma_pc
    self.experience_history_size = experience_history_size
    self.max_global_time_step = max_global_time_step
    self.action_size = Environment.get_action_size(env_type, env_name)
    
    self.local_network = UnrealModel(self.action_size,
                                     thread_index,
                                     use_pixel_change,
                                     use_value_replay,
                                     use_reward_prediction,
                                     pixel_change_lambda,
                                     entropy_beta,
                                     device)
    self.local_network.prepare_loss()

    self.apply_gradients = grad_applier.minimize_local(self.local_network.total_loss,
                                                       global_network.get_vars(),
                                                       self.local_network.get_vars())
    
    self.sync = self.local_network.sync_from(global_network)
    self.experience = Experience(self.experience_history_size)
    self.local_t = 0
    self.initial_learning_rate = initial_learning_rate
    self.episode_reward = 0.0
    self.episode_step = 0

    # For log output
    self.prev_local_t = 0

  def prepare(self):
    self.environment = Environment.create_environment(self.env_type,
                                                      self.env_name,
                                                      self.thread_index)

  def stop(self):
    self.environment.stop()
    
  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  
  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  
  def _record_score(self, sess, score_input, summary_op_score, summary_writer, score, global_t):
    print ('update tensorboard')
    summary_str = sess.run(summary_op_score, feed_dict={score_input: score})
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()

    
  def set_start_time(self, start_time):
    self.start_time = start_time


  def _fill_experience(self, sess):
    """
    Fill experience buffer until buffer is full.
    """
    # print ('fill exp')
    prev_state = self.environment.last_state
    last_action = self.environment.last_action
    last_reward = self.environment.last_reward
    last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                  self.action_size,
                                                                  last_reward)
    
    # print('testing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # self.local_network.run_path_laser(sess,self.environment.last_state)

    pi_, _ = self.local_network.run_base_policy_and_value(sess,
                                                          self.environment.last_state)
    action = self.choose_action(pi_)
    
    new_state, reward, terminal, info = self.environment.process(action)
    if info == 'f':
        return
    
    frame = ExperienceFrame(prev_state, reward, action, terminal, new_state, 
                            last_action, last_reward)
    self.experience.add_frame(frame)
    # print ('FILL EXP  thread: ', self.thread_index, 'experience size: ', len(self.experience._frames))

    if terminal:
      self.environment.reset()
    if self.experience.is_full():
      self.environment.reset()
      print("Replay buffer filled")


  def _print_log(self, global_t):
    if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
        global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))
    

  def _process_base(self, sess, global_t, score_input, summary_op_score, summary_writer):
    # [Base A3C]
    states = []
    last_action_rewards = []
    actions = []
    rewards = []
    values = []

    terminal_end = False

    # start_lstm_state = self.local_network.base_lstm_state_out

    # t_max times loop
    for _ in range(self.local_t_max):
      # Prepare last action reward
      last_action = self.environment.last_action
      last_reward = self.environment.last_reward
      last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                    self.action_size,
                                                                    last_reward)
      
      pi_, value_ = self.local_network.run_base_policy_and_value(sess,
                                                                 self.environment.last_state)
      
      
      action = self.choose_action(pi_)

      states.append(self.environment.last_state)
      last_action_rewards.append(last_action_reward)
      actions.append(action)
      values.append(value_)

      if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
        print("pi={}".format(pi_))
        print(" V={}".format(value_))

      prev_state = self.environment.last_state

      # Process game
      new_state, reward, terminal, info = self.environment.process(action)
      if self.episode_step > 50:
        terminal = True
      if info == 'f':
        continue;

      frame = ExperienceFrame(prev_state, reward, action, terminal, new_state, 
                              last_action, last_reward)

      # Store to experience
      self.experience.add_frame(frame)
      # print ('PROCESS BASE thread: ', self.thread_index, action, 'experience size: ', len(self.experience._frames))

      self.episode_reward += reward
      self.episode_step += 1

      rewards.append( reward )

      self.local_t += 1

      if terminal:
        terminal_end = True
        print("thread {} score={:f}".format(self.thread_index, self.episode_reward))

        self._record_score(sess, score_input, summary_op_score, summary_writer, self.episode_reward, global_t)
          
        self.episode_reward = 0.0
        self.episode_step = 0
        self.environment.reset()
        # self.local_network.reset_state()
        break

    R = 0.0
    if not terminal_end:
      R = self.local_network.run_base_value(sess, new_state, frame.get_last_action_reward(self.action_size))

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_a = []
    batch_adv = []
    batch_R = []

    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + self.gamma * R
      adv = R - Vi
      a = np.zeros([self.action_size])
      a[ai] = 1.0

      batch_si.append(si)
      batch_a.append(a)
      batch_adv.append(adv)
      batch_R.append(R)

    batch_si.reverse()
    batch_a.reverse()
    batch_adv.reverse()
    batch_R.reverse()

    return batch_si, last_action_rewards, batch_a, batch_adv, batch_R

  
  def _process_pc(self, sess):
    # [pixel change]
    # Sample 20+1 frame (+1 for last next state)
    pc_experience_frames = self.experience.sample_sequence(self.local_t_max+1)
    # pc_experience_frames = self.experience.sample_sequence(experience._history_size/2)
    # Revese sequence to calculate from the last
    pc_experience_frames.reverse()

    batch_pc_si = []
    batch_pc_a = []
    batch_pc_R = []
    batch_pc_last_action_reward = []
    
    pc_R = np.zeros([20,20], dtype=np.float32)
    if not pc_experience_frames[0].terminal:
      pc_R = self.local_network.run_pc_q_max(sess,
                                             pc_experience_frames[0].state,
                                             pc_experience_frames[0].get_last_action_reward(self.action_size))


    for frame in pc_experience_frames[1:]:
      pc_R = frame.pixel_change + self.gamma_pc * pc_R
      a = np.zeros([self.action_size])
      a[frame.action] = 1.0
      last_action_reward = frame.get_last_action_reward(self.action_size)
      
      batch_pc_si.append(frame.state)
      batch_pc_a.append(a)
      batch_pc_R.append(pc_R)
      batch_pc_last_action_reward.append(last_action_reward)

    batch_pc_si.reverse()
    batch_pc_a.reverse()
    batch_pc_R.reverse()
    batch_pc_last_action_reward.reverse()
    
    return batch_pc_si, batch_pc_last_action_reward, batch_pc_a, batch_pc_R

  
  def _process_vr(self, sess):
    # [Value replay]
    # Sample 20+1 frame (+1 for last next state)
    # vr_experience_frames = self.experience.sample_sequence(self.local_t_max+1)
    vr_experience_frames = self.experience.sample_sequence(int(self.experience._history_size/2))

    # Revese sequence to calculate from the last
    vr_experience_frames.reverse()

    batch_vr_si = []
    batch_vr_R = []
    batch_vr_last_action_reward = []

    vr_R = 0.0
    if not vr_experience_frames[0].terminal:
      vr_R = self.local_network.run_vr_value(sess,
                                             vr_experience_frames[0].state)
    
    # t_max times loop
    for frame in vr_experience_frames[1:]:
      vr_R = frame.reward + self.gamma * vr_R
      batch_vr_si.append(frame.state)
      batch_vr_R.append(vr_R)
      last_action_reward = frame.get_last_action_reward(self.action_size)
      batch_vr_last_action_reward.append(last_action_reward)

    batch_vr_si.reverse()
    batch_vr_R.reverse()
    batch_vr_last_action_reward.reverse()

    return batch_vr_si, batch_vr_last_action_reward, batch_vr_R

  
  def _process_rp(self):
    # [Reward prediction]
    rp_experience_frames = self.experience.sample_rp_sequence()
    # rp_experience_frames = self.experience.sample_sequence(self.local_t_max+1)

    # 4 frames

    batch_rp_si = []
    batch_rp_action = []
    batch_rp_reward = []
    
    for frame in rp_experience_frames[1:]:
        batch_rp_action.append ([frame.action])
        batch_rp_si.append(frame.state)
        rp_c = [0.0, 0.0, 0.0, 0.0]
        if frame.reward == 9:
              rp_c[0] = 1
        elif frame.reward == 1:
              rp_c[1] = 1
        elif frame.reward == -3:
              rp_c[2] = 1
        else:
              rp_c[3] = 1
        batch_rp_reward.append(rp_c)
   
    return batch_rp_si, batch_rp_action, batch_rp_reward
  
  
  def process(self, sess, global_t, 
              score_input, vr_loss_input, rp_loss_input, 
              summary_writer, summary_op_score, summary_op_loss):
    # Fill experience replay buffer
    if not self.experience.is_full():
      self._fill_experience(sess)
      return 0

    start_local_t = self.local_t

    cur_learning_rate = self._anneal_learning_rate(global_t)

    # Copy weights from shared to local
    sess.run( self.sync )

    # [Base]
    batch_si, batch_last_action_rewards, batch_a, batch_adv, batch_R = \
          self._process_base(sess,
                             global_t,
                             score_input, summary_op_score,
                             summary_writer)
    feed_dict = {
      self.local_network.base_input: batch_si,
      self.local_network.base_a: batch_a,
      self.local_network.base_adv: batch_adv,
      self.local_network.base_r: batch_R,  # true return
      # self.local_network.base_initial_lstm_state: start_lstm_state,
      # [common]
      self.learning_rate_input: cur_learning_rate
    }

    # [Pixel change]
    if self.use_pixel_change:
      batch_pc_si, batch_pc_last_action_reward, batch_pc_a, batch_pc_R = self._process_pc(sess)

      pc_feed_dict = {
        self.local_network.pc_input: batch_pc_si,
        self.local_network.pc_a: batch_pc_a,
        self.local_network.pc_r: batch_pc_R
      }
      feed_dict.update(pc_feed_dict)

    # [Value replay]
    if self.use_value_replay:
      batch_vr_si, batch_vr_last_action_reward, batch_vr_R = self._process_vr(sess)
      
      vr_feed_dict = {
        self.local_network.vr_input: batch_vr_si,
        self.local_network.vr_r: batch_vr_R  # predicted return using the network
      }
      feed_dict.update(vr_feed_dict)

    # [Reward prediction]
    if self.use_reward_prediction:
      batch_rp_si, batch_rp_action, batch_rp_reward = self._process_rp()

      rp_feed_dict = {
        self.local_network.rp_input: batch_rp_si,
        self.local_network.rp_action: batch_rp_action,
        self.local_network.rp_reward: batch_rp_reward
      }
      feed_dict.update(rp_feed_dict)

    # Calculate gradients and copy them to global netowrk.
    base_loss, _ = sess.run([self.local_network.base_loss,self.apply_gradients], feed_dict=feed_dict )
    # vr_loss = sess.run([self.local_network.vr_loss], feed_dict=feed_dict )

    # rp_loss = sess.run([self.local_network.rp_loss], feed_dict=feed_dict )
    
    self._print_log(global_t)
    
    if self.use_value_replay:
      base_loss, vr_loss = sess.run([self.local_network.base_loss, self.local_network.vr_loss], feed_dict=feed_dict )
      summary_str = sess.run(summary_op_loss, feed_dict={vr_loss_input: vr_loss})
      summary_writer.add_summary(summary_str, global_t)
      summary_writer.flush()

    # print ('update')
    if self.use_reward_prediction and self.use_value_replay:
      base_loss, vr_loss, rp_loss = sess.run([self.local_network.base_loss, 
                                                self.local_network.vr_loss, 
                                                self.local_network.rp_loss], feed_dict=feed_dict )
      summary_str = sess.run(summary_op_loss, feed_dict={vr_loss_input: vr_loss, rp_loss_input : rp_loss})
      summary_writer.add_summary(summary_str, global_t)
      summary_writer.flush()

    # Return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t
