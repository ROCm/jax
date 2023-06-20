# JAX Builds on ROCm
This directory contains files and setup instructions to build and test JAX for ROCm in Docker environment (runtime and CI). You can build, test and run JAX on ROCm yourself!
***
### Build JAX-ROCm in docker for the runtime

1.  Install Docker: Follow the [instructions on the docker website](https://docs.docker.com/engine/installation/).

2. Build a runtime JAX-ROCm docker container and keep this image by running the following command. This builds Python 3.9 container

    ./build/rocm/ci_build.sh --keep_image --py_version==3.9.0 --runtime bash -c "./build/rocm/build_rocm.sh"

3. To launch a JAX-ROCm container: If the build was successful, there should be a docker image with name "jax-rocm:latest" in list of docker images (use "docker images" command to list them).
```
sudo docker run -it --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --entrypoint /bin/bash jax-rocm:latest
```

***
### Build and Test JAX-ROCm in docker for CI jobs
This folder has all the scripts necessary to build and run tests for JAX-ROCm.
The following command will build JAX on ROCm and run all the tests inside docker (script should be called from JAX root folder).
```
./build/rocm/ci_build.sh
```


***
### JAX ROCm Releases
We strive to push all ROCm related changes to the OpenXLA repository. However, at times some JAX/JAXLIB changes for ROCm may not be present in upstream JAX repo.Therefore, we have ROCm Jax/Jaxlib branches that are associated with a Jaxlib release. These
are available in ROCm fork of JAX https://github.com/ROCmSoftwarePlatform/jax. See branches named as rocm-jaxlib-[jaxlib-version]. For examples, for jaxlib-v0.4.10, the branch is named rocm-jaxlib-v0.4.10. See path https://github.com/ROCmSoftwarePlatform/jax/tree/rocm-jaxlib-v0.4.10

JAX and Jaxlib wheels for ROCm are available here
```
https://github.com/ROCmSoftwarePlatform/jax/releases
```

***
### XLA for JAX ROCm
We strive to push all ROCm related changes to the OpenXLA repository. However, at times some XLA changes for ROCm may not be upstreamed to XLA repo.Therefore, we have ROCm XLA branches that are associated with a Jaxlib release. These are available in ROCm fork of XLA here https://github.com/ROCmSoftwarePlatform/xla. See branches named as rocm-jaxlib-[jaxlib version]. For example, for jaxlib-v0.4.10, the branch is named rocm-jaxlib-v0.4.10. See path https://github.com/ROCmSoftwarePlatform/xla/tree/rocm-jaxlib-v0.4.10
