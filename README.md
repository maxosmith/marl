# Multiagent Reinforcement Learning


## Docker

`docker build --target base -t marl -f Dockerfile.base .`

`docker run marl python /openspiel/open_spiel/python/examples/matrix_game_example.py`

`docker run -it --entrypoint /bin/bash marl`
