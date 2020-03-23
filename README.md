# upsidedown

Implementation of the Upside Down Reinforcement Learning (UDRL) based on the papers https://arxiv.org/abs/1912.02875 and https://arxiv.org/abs/1912.02877.

## Effectiveness

The CartPole should work but the LunarLander never learns completely.

## Setup

Create new environment:

    conda env create -f environment.yml
    conda activate upsidedown

Update existing project environment:

    conda env update -f environment.yml  --prune

Clone the openai gym repo and run:

    git clone git@github.com:openai/gym.git
    cd gym
    pip install -e '.[atari]'
