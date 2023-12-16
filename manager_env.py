import typing as t
import torch as th
import numpy as np
import vizdoom
from gymnasium import Env
from gymnasium import spaces
from stable_baselines3.common import vec_env
from action_combine import *
from stable_baselines3 import PPO
Frame = np.ndarray
from custom_cnn import CustomCNN
import pickle
#from image_classify import classify
#from new_train import CustomCNN


class DoomEnv(Env):
    """Wrapper environment following OpenAI's gym interface for a VizDoom game instance."""

    def __init__(self, game: vizdoom.DoomGame,frame_processor: t.Callable,frame_skip: int = 4,evl = False):
        super().__init__()

        # Determine action space
        self.game = game
        #self.ll = np.copy(ll)
        
        #self.prompts = prompts
        self.lb = []
        labels_path = "/Users/ericsunkuan/Desktop/NTUEE/112-1/RL/final_project/llm/hud/labels01_resize/"
        for i in range(100):       
            f0 = open(labels_path+f"{i}.pkl","rb")
            label = pickle.load(f0)
            label = np.array(label)
            #print(label)
            self.lb.append(label)
        '''
        self.possible_actions = get_available_actions(np.array([
             Button.MOVE_FORWARD, Button.MOVE_RIGHT, 
             Button.MOVE_LEFT, Button.TURN_LEFT, Button.TURN_RIGHT]))
        '''
        self.action_space = spaces.Discrete(2)

        # Determine observation space
        h, w, c = game.get_screen_height(), game.get_screen_width(), game.get_screen_channels()
        new_h, new_w, new_c = frame_processor(np.zeros((h, w, c))).shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(new_h, new_w, new_c), dtype=np.uint8)

        # Assign other variables
        self.item_count = 0
        #self.possible_actions = np.eye(self.action_space.n).tolist()  # VizDoom needs a list of buttons states.
        self.frame_skip = frame_skip
        self.frame_processor = frame_processor
        self.init_x, self.init_y = self._get_player_pos()
        self.last_x, self.last_y = self._get_player_pos()
        self.step_count = 0
        self.health = self.game.get_game_variable(vizdoom.GameVariable.HEALTH)
        self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.state = self.empty_frame
        self.nav_agent = PPO.load('/Users/ericsunkuan/Desktop/NTUEE/112-1/RL/final_project/rl_project-Jeff/new_navi.zip', custom_objects={'policy_kwargs': {'features_extractor_class': CustomCNN}})
        #self.nav_agent = PPO.load('/Users/ericsunkuan/Desktop/NTUEE/112-1/RL/final_project/rl_project-Jeff/new_navi.zip', custom_objects={'policy_kwargs':{'feature_extractor_class':CustomCNN}})
        self.shooting_agent = PPO.load('/Users/ericsunkuan/Desktop/NTUEE/112-1/RL/final_project/rl_project-Jeff/new_shooting.zip', custom_objects={'policy_kwargs':{'features_extractor_class':CustomCNN}})
        self.nav_actions = get_available_actions(np.array([
             Button.MOVE_FORWARD, Button.MOVE_RIGHT, 
             Button.MOVE_LEFT, Button.TURN_LEFT, Button.TURN_RIGHT]))
        self.shooting_actions = get_available_actions(np.array([
             Button.MOVE_FORWARD, Button.MOVE_RIGHT, 
             Button.MOVE_LEFT, Button.TURN_LEFT, Button.TURN_RIGHT, Button.ATTACK]))
        self.kill_count =0
        self.w = 0
        self.prompt_done = 0
        self.evl = evl
        self.record = []
    def recording(self):
        if not self.evl:
            return
        if self.game.is_episode_finished():
            return
        else:
            state_o = self.game.get_state().screen_buffer
            self.record.append(np.moveaxis(state_o,0,-1))



    def step(self, action: int,) -> t.Tuple[Frame, int, bool, t.Dict]:
        self.w+=1
        if self.prompt_done >= 100000:
            self.prompt_done = -1
        elif self.prompt_done > -1:
            if self.game.get_state() is not None:
                state = self.game.get_state()
                label_goal = state.labels_buffer
                
                for i in range(100):
                    print(np.sum(self.lb[i]))
                target_label = 0
                target_label = self.classify(label_goal)
                #target_prompt = self.prompts[target_label]
                target_prompt = 0
            prompt_r = self.prompt_reward(self.prompt_done,target_prompt,action)
            self.prompt_done += 1
            

        if self.prompt_done == -1 :   ### check if the 5 skills of last prompt has been evaluated
            if self.game.get_state() is not None:
                state = self.game.get_state()
                slabels = state.labels
                Imp_count=0
                for label in slabels:
                    if label.object_name == 'DoomImp' and label.value >= 50:
                        Imp_count += 1
                if Imp_count > 0 :     ### if last prompt is evaluated and there is enemy in vision --> prompt
                    self.prompt_done = 0
                    label_goal = state.labels_buffer
                    
                    target_label = self.classify(label_goal)
                    target_label = 0
                    #target_prompt = self.prompts[target_label]
                    target_prompt = 0
                    


        self.recording()
        if action == 0:
            reward,self.state,done = self.navigation()
        elif action == 1:
            reward,self.state,done = self.shooting()
        '''    
        print(self.w%10)
        print(f'action:{action}')
        print(f'reward:{reward}')
        '''
        info ={'info':[]}
        if done and self.evl:
            info = {'info':self.record}
        return self.state, reward, done, False, info

    def classify(self,label_goal):
        p = np.full(100, 10**6)
        p1 = th.full((100,), 10**6)
        for i in range(label_goal.shape[0]):
            for j in range(label_goal.shape[1]):
                if(label_goal[i][j]<0 or label_goal[i][j]>1):
                    label_goal[i][j] = 0
            for i in range(100):
        #     print(np.sum(labels[i]))
                p[i]=np.sum(np.abs(label_goal,self.get_labels()[i]))
                #lg = th.tensor(label_goal)
                #p1[i] = th.sum(th.abs(lg-self.lb))
                #p[i]=total_difference = sum(sum(abs(a - b) for a, b in zip(row1, row2)) for row1, row2 in zip(label_goal, self.get_labels()[i]))
            #p[i] = total_absolute_difference(label_goal,labels[i])
        p = th.tensor(p)
        # print(p)
        # print(p.argmin())
        # print(p[p.argmin()])
        return p1.argmin()    
    
    def get_labels(self):
        return self.lb
    
    def prompt_reward(self,prompt_num,prompt,action):
        
        

        
        return 0
    
    
    def navigation(self,evl=False):
        reward = 0
        action,_ = self.nav_agent.predict(self.state)
        reward += self.game.make_action(self.nav_actions[action],4)
        reward += self.reward(action)
        self.recording()
        while True:
            done = self.game.is_episode_finished()
            self.state = self._get_frame(done)
            if done :
                return reward, self.state, done

            state = self.game.get_state()
            labels = state.labels
            Imp_count=0
            for label in labels:
                if label.object_name == 'DoomImp' and label.value >= 200:
                    Imp_count += 1
            if Imp_count > 0 :
                #print('kkkk')
                return reward, self.state, done
            else:
                action,_ = self.nav_agent.predict(self.state)
                reward += self.game.make_action(self.nav_actions[action],4)
                self.recording()
                reward += self.reward(action)
    def shooting(self):
        reward = 0
        no_e_count = 0
        
        while True:
            done = self.game.is_episode_finished()
            self.state = self._get_frame(done)
            if done :
                return reward, self.state, done
            
            state = self.game.get_state()
            labels = state.labels
            Imp_count=0
            for label in labels:
                if label.object_name == 'DoomImp' and label.value >= 200:
                    Imp_count += 1
            no_e_count += 1 if Imp_count == 0 else 0
            no_e_count = 0 if Imp_count>0 else no_e_count
            if Imp_count > 0 or no_e_count <=3 :
                action,_ = self.shooting_agent.predict(self.state)
                reward += self.game.make_action(self.shooting_actions[action],4)
                self.recording()
                reward += self.reward(action) 
            else:
                return reward, self.state, done
    
    def reward(self,action):
        reward=0
        kill_count = self.game.get_game_variable(vizdoom.GameVariable.KILLCOUNT)
        reward += max(kill_count-self.kill_count,0)
        self.kill_count = kill_count
        return reward
    def _get_player_pos(self):
        """Returns the player X- and Y- coordinates."""
        return self.game.get_game_variable(vizdoom.GameVariable.POSITION_X), self.game.get_game_variable(
                            vizdoom.GameVariable.POSITION_Y)
    def reset(self,seed=None) -> Frame:
        """Resets the environment.

        Returns:
                        The initial state of the new environment.
        """
        self.game.new_episode()
        self.state = self._get_frame()
        self.step_count = 0
        self.kill_count = 0
        self.record = []
        self.last_x, self.last_y = self._get_player_pos()
        self.health = self.game.get_game_variable(vizdoom.GameVariable.HEALTH)
        return self.state,{}

    def close(self) -> None:
        self.game.close()

    def render(self, mode='human'):
        pass

    def _get_frame(self, done: bool = False) -> Frame:
        return self.frame_processor(self.game.get_state().screen_buffer) if not done else self.empty_frame

    

class DoomWithBots(DoomEnv):

    def __init__(self, game, frame_processor, frame_skip, n_bots):
        super().__init__(game, frame_processor, frame_skip)
        self.n_bots = n_bots
        self.last_frags = 0
        self._reset_bots()

        # Redefine the action space using combinations.
        self.possible_actions = get_available_actions(np.array(game.get_available_buttons()))
        self.action_space = spaces.Discrete(len(possible_actions))

    def step(self, action):
        self.game.make_action(self.possible_actions[action], self.frame_skip)

        # Compute rewards.
        frags = self.game.get_game_variable(GameVariable.FRAGCOUNT)
        reward = frags - self.last_frags
        self.last_frags = frags

        # Check for episode end.
        self._respawn_if_dead()
        done = self.game.is_episode_finished()
        self.state = self._get_frame(done)

        return self.state, reward, done, {}

    def reset(self,seed = None):
        self._reset_bots()
        self.last_frags = 0

        return super().reset()

    def _respawn_if_dead(self):
        if not self.game.is_episode_finished():
            if self.game.is_player_dead():
                self.game.respawn_player()

    def _reset_bots(self):
                # Make sure you have the bots.cfg file next to the program entry point.
        self.game.send_game_command('removebots')
        for i in range(self.n_bots):
            self.game.send_game_command('addbot')


def create_env(scenario: str, **kwargs) -> DoomEnv:
        # Create a VizDoom instance.
    game = vizdoom.DoomGame()
    game.load_config(f'/Users/ericsunkuan/Desktop/NTUEE/112-1/RL/final_project/rl_project-Jeff/scenarios/{scenario}.cfg')
    game.set_window_visible(True)
    game.set_death_penalty(2)
    game.set_labels_buffer_enabled(True)
    game.init()

    # Wrap the game with the Gym adapter.
    return DoomEnv(game, **kwargs)


def create_env_with_bots(scenario, **kwargs) -> DoomEnv:
        # Create a VizDoom instance.
    game = vizdoom.DoomGame()
    game.load_config(f'scenarios/{scenario}.cfg')
    game.add_game_args('-host 1 -deathmatch +viz_nocheat 0 +cl_run 1 +name AGENT +colorset 0' +
                                   '+sv_forcerespawn 1 +sv_respawnprotect 1 +sv_nocrouch 1 +sv_noexit 1')
    game.set_window_visible(False)
    game.init()

    return DoomWithBots(game, **kwargs)


def create_vec_env(n_envs=1, **kwargs) -> vec_env.VecTransposeImage:
    return vec_env.VecTransposeImage(vec_env.DummyVecEnv([lambda: create_env(**kwargs)] * n_envs))


def vec_env_with_bots(n_envs=1, **kwargs) -> vec_env.VecTransposeImage:
    return vec_env.VecTransposeImage(vec_env.DummyVecEnv([lambda: create_env_with_bots(**kwargs)] * n_envs))


def create_eval_vec_env(**kwargs) -> vec_env.VecTransposeImage:
    return create_vec_env(n_envs=1, **kwargs)

