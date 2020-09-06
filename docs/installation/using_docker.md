# Installation using Docker

## Install Docker

Make sure `docker` has been installed. You can refer to the [official tutorial](https://docs.docker.com/install/).

Install NVIDIA Container Toolkit if using GPU. You can refer to [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

## Docker image

You can build image locally or using pre-build images,  pre-built Docker images are available on DockerHub.

### Pull Docker Image

You can directly pull the pre-build docker images for athena. We have created the following docker images:

[athena-tf-2.1.0-gpu-py3](https://hub.docker.com/repository/docker/garygao99/athena/)

Download the image as below:
```
docker pull garygao99/athena:tf-2.1.0-gpu-py3
```

### Build images

You can build image locally as below:
```
docker build -t garygao99/athena:tf-2.1.0-gpu-py3 .
```

## Create Container

After the image downloaded, create a container.

```bash
docker run -it --gpus all garygao99/athena:tf-2.1.0-gpu-py3 /bin/bash
```
