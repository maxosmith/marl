# Multiagent Reinforcement Learning


## Docker

`docker build --target base -t marl -f Dockerfile.base .`

`docker run marl python /openspiel/open_spiel/python/examples/matrix_game_example.py`

`docker run -it --entrypoint /bin/bash marl`


The Docker build files and directions are largely taken from `Launchpad`

Build the Docker container to be used for compiling and testing. You can specify `tensorflow_pip` parameter to set the version of Tensorflow to build against. You can also specify which version(s) of Python container should support. The command below enables support for Python 3.7, 3.8, 3.9 and 3.10.
```bash
docker build --tag marl -f docker/build.dockerfile .

```
```bash
docker build --tag marl --platform linux/amd64 -f docker/build.dockerfile .
```

The next step is to enter the built Docker image, binding sources to `/tmp/` within the container.
```bash
docker run --rm --mount "type=bind,src=$PWD,dst=/tmp/marl" \
  -it marl bash
```

```bash
/tmp/launchpad/oss_build.sh --tf_package "tensorflow==2.9.*" --reverb_package "dm-reverb==0.7.2"
```
