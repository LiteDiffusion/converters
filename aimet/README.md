
## Build Docker Images

1. Clone AIMET repository.

```bash
git clone https://github.com/quic/aimet.git
cd aimet/Jenkins/
```

2. Build Docker image.

```bash
docker build . -f Dockerfile.torch-cpu \
    -t artifacts.codelinaro.org/codelinaro-aimet/aimet-dev:1.25.0.torch-cpu \
    --build-arg http_proxy=http://127.0.0.1:7890 \
    --build-arg https_proxy=http://127.0.0.1:7890 \
    --network=host
```

If you want to build the image for GPU, use the following command.

```bash
docker build . -f Dockerfile.torch-gpu \
    -t artifacts.codelinaro.org/codelinaro-aimet/aimet-dev:1.25.0.torch-gpu \
    --build-arg http_proxy=http://127.0.0.1:7890 \
    --build-arg https_proxy=http://127.0.0.1:7890 \
    --network=host
```

## Run Docker Container

```bash
docker run --rm --name=aimet-dev-torch-cpu -v $PWD:$PWD -w $PWD -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro --network=host --ulimit core=-1 --ipc=host --shm-size=8G --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -it artifacts.codelinaro.org/codelinaro-aimet/aimet-dev:1.25.0.torch-cpu
```

If you want to run the container for GPU, use the following command.

```bash
docker run --rm --gpus all --name=aimet-dev-torch-gpu -v $PWD:$PWD -w $PWD -v /etc/localtime:/etc/localtime:ro -v /etc/timezone:/etc/timezone:ro --network=host --ulimit core=-1 --ipc=host --shm-size=8G --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
-it zhenghaoz/aimet-dev:1.25.0.torch-gpu
```
