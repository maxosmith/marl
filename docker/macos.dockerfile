# docker build --platform linux/amd64 --tag marl -f docker/macos2.dockerfile . --no-cache
# docker run --platform linux/amd64 --rm -it marl bash
#
# https://github.com/ethanluoyc/launchpad/releases
# https://github.com/ethanluoyc/launchpad/blob/build-fix-v0.6.0/docker/build.dockerfile
FROM --platform=linux/amd64 tensorflow/build:latest-python3.9

ARG APT_COMMAND="apt-get -o Acquire::Retries=3 -y"

RUN ${APT_COMMAND} update && ${APT_COMMAND} install -y --no-install-recommends \
  software-properties-common \
  aria2 \
  build-essential \
  curl \
  git \
  less \
  libfreetype6-dev \
  libhdf5-serial-dev \
  libpng-dev \
  libzmq3-dev \
  lsof \
  pkg-config \
  python3.8-dev \
  python3.9-dev \
  python3.10-dev \
  python3.11-dev \
  # python >= 3.8 needs distutils for packaging.
  python3.8-distutils \
  python3.9-distutils \
  python3.10-distutils \
  python3.11-distutils \
  rename \
  rsync \
  sox \
  unzip \
  vim \
  && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Installs known working version of bazel.
ARG bazel_version=4.1.0
ENV BAZEL_VERSION ${bazel_version}
RUN mkdir /bazel && \
  cd /bazel && \
  curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
  chmod +x bazel-*.sh && \
  ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
  cd / && \
  rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

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


# https://github.com/deepmind/launchpad/issues/30
RUN mkdir -p $CONDA_PREFIX/etc/conda/activate.d
RUN mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
RUN echo 'unset LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

# Install jax dependencies.
RUN pip install --upgrade pip
RUN pip install numpy wheel build

RUN conda run -n marl conda install -c conda-forge jaxlib
RUN conda run -n marl conda install -c conda-forge jax
RUN conda run -n marl conda install tensorflow

# Removes existing links so they can be created to point where we expect.
RUN rm /dt9/usr/include/x86_64-linux-gnu/python3.8
RUN rm /dt9/usr/include/x86_64-linux-gnu/python3.9
RUN rm /dt9/usr/include/x86_64-linux-gnu/python3.10
RUN rm /dt9/usr/include/x86_64-linux-gnu/python3.11

# Needed until this is included in the base TF image.
RUN ln -s "/usr/include/x86_64-linux-gnu/python3.8" "/dt9/usr/include/x86_64-linux-gnu/python3.8"
RUN ln -s "/usr/include/x86_64-linux-gnu/python3.9" "/dt9/usr/include/x86_64-linux-gnu/python3.9"
RUN ln -s "/usr/include/x86_64-linux-gnu/python3.10" "/dt9/usr/include/x86_64-linux-gnu/python3.10"
RUN ln -s "/usr/include/x86_64-linux-gnu/python3.11" "/dt9/usr/include/x86_64-linux-gnu/python3.11"


WORKDIR /reverb/
RUN git clone https://github.com/deepmind/reverb.git .
# https://github.com/deepmind/reverb/pull/128
# https://github.com/deepmind/reverb/pull/119
RUN git fetch origin pull/119/head:branch
RUN ln -s /usr/lib/x86_64-linux-gnu/libpython3.9.so /opt/conda/envs/marl/lib/python3.9/config-3.9-x86_64-linux-gnu/libpython3.9.so
RUN bazel build -c opt --copt=-mavx --config=manylinux2014 --test_output=errors //...
RUN ./bazel-bin/reverb/pip_package/build_pip_package \ --dst /tmp/reverb_build/dist/
RUN pip install --upgrade /tmp/reverb_build/dist/*.whl

# Update binutils to avoid linker(gold) issue. See b/227299577#comment9
RUN \
  wget http://old-releases.ubuntu.com/ubuntu/pool/main/b/binutils/binutils_2.35.1-1ubuntu1_amd64.deb \
  && wget http://old-releases.ubuntu.com/ubuntu/pool/main/b/binutils/binutils-x86-64-linux-gnu_2.35.1-1ubuntu1_amd64.deb \
  && wget http://old-releases.ubuntu.com/ubuntu/pool/main/b/binutils/binutils-common_2.35.1-1ubuntu1_amd64.deb \
  && wget http://old-releases.ubuntu.com/ubuntu/pool/main/b/binutils/libbinutils_2.35.1-1ubuntu1_amd64.deb

RUN \
  dpkg -i binutils_2.35.1-1ubuntu1_amd64.deb \
  binutils-x86-64-linux-gnu_2.35.1-1ubuntu1_amd64.deb \
  binutils-common_2.35.1-1ubuntu1_amd64.deb \
  libbinutils_2.35.1-1ubuntu1_amd64.deb

WORKDIR /
# RUN pip install dm-reverb==0.10.0
# RUN pip install https://github.com/ethanluoyc/launchpad/releases/download/v0.6.0rc0/dm_launchpad-0.6.0rc0-cp310-cp310-manylinux2014_x86_64.whl
