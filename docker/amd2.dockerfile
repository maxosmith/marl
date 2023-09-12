FROM --platform=linux/amd64 nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install base utilities
RUN apt-get update && \
  apt-get install -y sudo

# Install developer tools
RUN apt-get install -y build-essential && \
  apt-get install -y wget && \
  apt-get install -y git && \
  apt-get install -y g++ && \
  apt-get install -y python3-dev && \
  apt-get install -y python3-pip

# Install LaTeX build system
RUN apt-get install -y texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra texlive-xetex latexmk xindy dvipng ghostscript cm-super

# Clean apt
RUN sudo apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
  /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda update -y conda
RUN conda init

# Setup Environment
COPY docker/build_env.yml .
RUN conda env create -f build_env.yml
RUN echo "source activate marl" > ~/.bashrc
ENV PATH /opt/conda/envs/marl/bin:$PATH

COPY docker/reqs.txt .
RUN pip install -r reqs.txt


# Install JAX
# RUN conda run -n marl conda install -c conda-forge jaxlib
# RUN conda run -n marl conda install -c conda-forge jax
# RUN conda run -n marl conda install tensorflow
# RUN conda install tensorflow==2.8.2
# RUN pip install -U pip
# RUN pip install dm-launchpad[reverb]

# RUN pip install dm-launchpad[reverb]

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/marl/lib

# Installs known working version of bazel.
# ARG bazel_version=5.3.0
# ENV BAZEL_VERSION ${bazel_version}
# RUN mkdir /bazel && \
#   cd /bazel && \
#   curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
#   chmod +x bazel-*.sh && \
#   ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
#   cd / && \
#   rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# WORKDIR /launchpad
# RUN git clone https://github.com/deepmind/launchpad.git .
# ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/conda/envs/marl/lib
# RUN python configure.py
# RUN bazel build -c opt --copt=-mavx --config=manylinux2014 --test_output=errors //...
# RUN ./oss_build.sh --python 3.10
