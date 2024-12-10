# JAX Builds on ROCm
This directory contains files and setup instructions to build and test JAX for ROCm in a Docker environment (runtime and CI). You can build, test, and run JAX on ROCm yourself!
***
## JAX ROCm Releases

### Overview
We aim to push all ROCm-related changes to the OpenXLA repository. However, there may be times when certain JAX/jaxlib updates for ROCm are not yet reflected in the upstream JAX repository.

To address this, we maintain ROCm-specific JAX/jaxlib branches tied to JAX releases.
These branches are hosted in the ROCm fork of JAX and XLA:
* https://github.com/ROCm/jax
* https://github.com/ROCm/xla

### Branch Naming Convention
Branches are named in the format rocm-jaxlib-[jaxlib-version]. For example:
* For JAX version 0.4.35, the corresponding branch is `rocm-jaxlib-v0.4.35`.
You can access it at: https://github.com/ROCm/jax/tree/rocm-jaxlib-v0.4.35.

### Latest JAX Releases for ROCm

GitHub Releases:

```Bash
https://github.com/ROCm/jax/releases
```

Docker Images:
Prebuilt ROCm JAX Docker images are available on Docker Hub:
```Bash
https://hub.docker.com/r/rocm/jax-community/tags
```

PyPI Installation:
JAX can also be installed via PyPI using the following command:
```Bash
pip install jax[rocm]
```

Note: Earlier versions of jaxlib for ROCm are available on PyPI [jaxlib-rocm PyPI History](https://pypi.org/project/jaxlib-rocm/#history).

***
## Build JAX-ROCm in docker for the runtime

1.  Install Docker: Follow the [instructions on the docker website](https://docs.docker.com/engine/installation/).

2. Build a runtime JAX-ROCm docker container and keep this image by running the following command. Note: must pass in appropriate options. The example below builds Python 3.12 container.

```Bash
./build/rocm/ci_build.sh --py_version 3.12
```

3. To launch a JAX-ROCm container: If the build was successful, there should be a docker image with name "jax-rocm:latest" in list of docker images (use "docker images" command to list them).

```Bash
docker run -it -d --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 64G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v ./:/jax --name rocm_jax jax-rocm:latest /bin/bash
```
