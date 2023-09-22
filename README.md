# Multiagent Reinforcement Learning
Library of modules/blueprints commonly prevalent in (multiagent) reinforcement learning.
The goal of this project is to provide access to these independent modules without significant layers of engineering ontop of them that may obfuscate their interactions.
Thereby enabling practioners to adopt a _copy-modify-compose_ design methodology, facilitating rapid development of new algorithms..


## Installation

### Linux

```bash
# Set-up environment.
conda create -n py39
conda activate py39
conda install python=3.9 -c conda-forge
pip install requirements/linux.txt
pip install dm-reverb dm-launchpad
```

### Mac OS
Setting up LaunchPad for local usage is possible thanks largely to the effort of @cemlyn007.
```bash
# Set-up environment.
conda create -n py39
conda activate py39
conda install python=3.9 -c conda-forge
pip install requirements/mac.txt

# Install Bazelisk to build packages.
go get github.com/bazelbuild/bazelisk
go install github.com/bazelbuild/bazelisk@latest
export PATH=$PATH:$(go env GOPATH)/bin

# Install Reverb.
git clone https://github.com/google-deepmind/reverb
cd reverb
gh pr checkout 128
./oss_build.sh --release
cd ..

# Install LaunchPad.
git clone https://github.com/google-deepmind/launchpad
cd launchpad
gh pr checkout 40
./oss_build.sh --release
cd ..
```

There is one sharp-bit about this currently patchy solution to running on MacOS.
Tensorflow needs to be imported _before_ LaunchPad.
Please see discussion at https://github.com/google-deepmind/reverb/pull/128.


### Developer
Developers should additionally install the packages found in `requirements/mac.txt`.
