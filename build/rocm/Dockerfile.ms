################################################################################
ARG BASE_DOCKER=ubuntu:20.04
FROM $BASE_DOCKER as rt_build
################################################################################

# Add target file to help determine which device(s) to build for
ARG GPU_DEVICE_TARGETS="gfx900 gfx906 gfx908 gfx90a gfx940 gfx941 gfx942 gfx1030 gfx1100"
ENV GPU_DEVICE_TARGETS=${GPU_DEVICE_TARGETS}

# Install ROCM
ARG ROCM_VERSION=6.0.0
ARG CUSTOM_INSTALL
ARG ROCM_PATH=/opt/rocm-${ROCM_VERSION}
ENV ROCM_PATH=${ROCM_PATH}
COPY ${CUSTOM_INSTALL} /${CUSTOM_INSTALL}
COPY setup.rocm.sh /setup.rocm.sh
RUN /setup.rocm.sh $ROCM_VERSION

# Set up paths
ENV HCC_HOME=$ROCM_PATH/hcc
ENV HIP_PATH=$ROCM_PATH/
ENV OPENCL_ROOT=$ROCM_PATH/opencl
ENV PATH="$HCC_HOME/bin:$HIP_PATH/bin:${PATH}"
ENV PATH="$ROCM_PATH/bin:${PATH}"
ENV PATH="$OPENCL_ROOT/bin:${PATH}"
ENV PATH="/root/bin:/root/.local/bin:$PATH"


# Install pyenv with different python versions
ARG PYTHON_VERSION=3.10.0
RUN git clone https://github.com/pyenv/pyenv.git /pyenv
ENV PYENV_ROOT /pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv install $PYTHON_VERSION
RUN eval "$(pyenv init -)" && pyenv local ${PYTHON_VERSION} && pip3 install --upgrade --force-reinstall setuptools pip && pip install numpy setuptools build wheel six auditwheel scipy==1.12.0 pytest pytest-rerunfailures matplotlib absl-py flatbuffers hypothesis

################################################################################
ARG BASE_DOCKER=ubuntu:20.04
FROM $BASE_DOCKER as ci_build
################################################################################
ARG ROCM_BUILD_JOB=ubuntu
ARG LKG_BUILD_NUM=main
ARG AMDGPU_CORE=""
ARG ROCM_VERSION=""
ARG ROCM_MAJ_MIN=""
ARG ROCM_PATH=""
#ENV ROCM_PATH=${ROCM_PATH}
ARG PYTHON_VERSION=3.9.0
COPY UTC* /usr/share/zoneinfo/UTC
RUN ln -sf /usr/share/zoneinfo/UTC /etc/localtime ; echo 'UTC' > /etc/timezone

RUN apt-get update --allow-insecure-repositories && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  software-properties-common \
  clang-6.0 \
  clang-format-6.0 \
  curl \
  g++-multilib \
  git \
  vim \
  libnuma-dev \
  virtualenv \
  python3-pip \
  pciutils \
  python-is-python3 \
  libffi-dev \
  libssl-dev \
  build-essential \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev curl \
  libncursesw5-dev \
  xz-utils \
  tk-dev \
  libxml2-dev \
  libxmlsec1-dev \
  libffi-dev \
  liblzma-dev \
  wget && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Add to get ppa
RUN apt-get update
RUN apt-get install -y software-properties-common

RUN apt update && apt install -y wget && wget http://artifactory-cdn.amd.com/artifactory/list/amdgpu-deb/amdgpu-install-internal_${ROCM_MAJ_MIN}-20.04-1_all.deb \
    && apt install -y ./amdgpu-install-internal_${ROCM_MAJ_MIN}-20.04-1_all.deb && apt update
RUN amdgpu-repo --amdgpu-build=${AMDGPU_CORE} --rocm-build=${ROCM_BUILD_JOB}/${LKG_BUILD_NUM} \
    && amdgpu-install --no-dkms --usecase=rocm -y
RUN apt install -y rocm-dev rocm-libs

ENV HCC_HOME=$ROCM_PATH/hcc
ENV HIP_PATH=$ROCM_PATH/hip
ENV OPENCL_ROOT=$ROCM_PATH/opencl
ENV PATH="$HCC_HOME/bin:$HIP_PATH/bin:${PATH}"
ENV PATH="$ROCM_PATH/bin:${PATH}"
ENV PATH="$OPENCL_ROOT/bin:${PATH}"

# Add target file to help determine which device(s) to build for
RUN bash -c 'echo -e "gfx900\ngfx906\ngfx908\ngfx90a\ngfx940\ngfx941\ngfx942\ngfx1030\ngfx1100" >> ${ROCM_PATH}/bin/target.lst'

# Need to explicitly create the $ROCM_PATH/.info/version file to workaround what seems to be a bazel bug
# The env vars being set via --action_env in .bazelrc and .tf_configure.bazelrc files are sometimes
# not getting set in the build command being spawned by bazel (in theory this should not happen)
# As a consequence ROCM_PATH is sometimes not set for the hipcc commands.
# When hipcc incokes hcc, it specifies $ROCM_PATH/.../include dirs via the `-isystem` options
# If ROCM_PATH is not set, it defaults to /opt/rocm, and as a consequence a dependency is generated on the
# header files included within `/opt/rocm`, which then leads to bazel dependency errors
# Explicitly creating the $ROCM_PATH/.info/version allows ROCM path to be set correrctly, even when ROCM_PATH
# is not explicitly set, and thus avoids the eventual bazel dependency error.
# The bazel bug needs to be root-caused and addressed, but that is out of our control and may take a long time
# to come to fruition, so implementing the workaround to make do till then
# Filed https://github.com/bazelbuild/bazel/issues/11163 for tracking this
RUN touch ${ROCM_PATH}/.info/version

ENV PATH="/root/bin:/root/.local/bin:$PATH"

RUN git clone https://github.com/pyenv/pyenv.git /pyenv
ENV PYENV_ROOT /pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv install $PYTHON_VERSION
RUN eval "$(pyenv init -)" && pyenv local ${PYTHON_VERSION} && pip3 install --upgrade --force-reinstall setuptools pip && pip install numpy setuptools build wheel six auditwheel scipy==1.12.0 pytest pytest-rerunfailures matplotlib absl-py

CMD [ "/bin/bash" ]
