# Collaboration and Competition

In this project you will train two reinforcement learning agents to play tennis using the **Unity Tennis** environment.

## The Environment

![Image](tennis_image.png)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Install dependencies

To run this project, you must have Python 3.6, Pytorch and Unity ML-Agents toolkit installed. Follow the instructions in Udacity's Deep Reinforcement Learning Nanodegree [repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install the required dependencies.

You must also use one of the following Unity environments for this project. Download the environment specific to your platform and place it in the same folder as this project. Unzip the file and extract the contents in the same folder.

* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## Running the code

You can run the code from any Python IDE or a terminal.

* To run the code in a Jupyter notebook, open the **Tennis.ipynb** notebook. This notebook goes through starting the Unity environment, creating the agent, training and simulation.
* To run from a terminal, run **tennis_train.py** to train the agent, or **tennis_sim.py** to watch a trained agent.

<pre><code>python tennis_train.py
python tennis_sim.py
</code></pre>

***Important***
For both the above steps, you must change the **file_name** variable in the code to match the appropriate platform.

## Further reading

**Report.ipynb** contains explanation on the training algorithm, hyperparameters, implementation and summarizes what you may expect in the training process.

*Ref: Deep Reinforcement Learning Nanodegree resources.*
