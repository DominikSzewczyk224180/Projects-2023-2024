from stable_baselines3 import PPO
import gym
import time
import os
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from ot2_gym_wrapper import OT2Env
import tensorboard

from clearml import Task

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Use the appropriate project name and task name (if you are in the first group in Dean's mentor group, use the project name 'Mentor Group D/Group 1')
# It can also be helpful to include the hyperparameters in the task name
task = Task.init(project_name='Mentor Group J/Group 2', task_name='224180 1')
#copy these lines exactly as they are
#setting the base docker image

#copy these lines exactly as they are
#setting the base docker image
task.set_base_docker('deanis/2023y2b-rl:latest')
#setting the task to run remotely on the default queue
task.execute_remotely(queue_name="default")


env = OT2Env()

#arguments
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)

args, unknown = parser.parse_known_args()



if unknown:
    print(f"Warning: Unknown arguments {unknown}")

#WandB
os.environ['WANDB_API_KEY'] = '9d863b8e66f3f7aa6399deebfc7c8a4851f677e5'
run = wandb.init(project="sb3_pendulum_demo", sync_tensorboard=True)

#model
model = PPO('MlpPolicy', env, verbose=1,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            tensorboard_log=f"runs/{run.id}",
            )

#callback
os.makedirs(f"models/{run.id}", exist_ok=True)
wandb_callback = WandbCallback(model_save_freq=1000,
                               model_save_path=f"models/{run.id}",
                               verbose=2,
                               )

# Train the model
timesteps = 100000
model.learn(total_timesteps=timesteps, callback=wandb_callback, progress_bar=True, tb_log_name=f"runs/{run.id}")

for i in range(10):
    model.learn(total_timesteps=timesteps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,
                tb_log_name=f"runs/{run.id}")
    model.save(f"models/{run.id}/{timesteps*(i+1)}")

# python "2. Robotics\Y2B-2023-OT2_Twin-main\task_11.py" --learning_rate 0.0001 --batch_size 64 --n_steps 2048 --n_epochs 10

# python "2. Robotics\Y2B-2023-OT2_Twin-main\task_11.py" --learning_rate 0.001 --batch_size 64 --n_steps 1024 --n_epochs 10
    
# python "2. Robotics\Y2B-2023-OT2_Twin-main\task_11.py" --learning_rate 0.001 --batch_size 32 --n_steps 1024 --n_epochs 15
