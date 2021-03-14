# %%
from IPython.display import clear_output
import tensorflow as tf
from vizdoom import *
from agent import Agent
import vizdoom as vzd
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings  # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' ## UNCOMMENT ME IF YOU GET OUT OF MEMORY ERROR


def create_environment():
    game = vzd.DoomGame()

    # Load the game
    game.set_doom_scenario_path("basic.wad")
    game.load_config("./basic.cfg")

    # Possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    game.set_window_visible(True)  # Render game
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.init()

    return game, possible_actions


game, possible_actions = create_environment()
n_actions = 3
learning_rate = 5e-4
max_steps = 100
gamma = 0.95
EPOCHS = 6001
agent = Agent(num_actions=n_actions, optimizer=tf.optimizers.RMSprop(learning_rate=learning_rate),
              env=game, gamma=gamma)
# agent.load_model('./model') ## Load last model
running_reward = 0
running_reward_hist = []
# %% Train!
for e in tqdm(range(EPOCHS)):
    reward = agent.train_step(max_steps)
    running_reward = reward*0.01 + running_reward*.99
    running_reward_hist.append(running_reward)
    clear_output(wait=True)
    if e % 500 == 0:
        agent.model.save('./model')  # Save model every 500
    print(f"Episode {e}, reward:{reward} running reward:{running_reward}")
    plt.plot(running_reward_hist)
    plt.show()

# %%
