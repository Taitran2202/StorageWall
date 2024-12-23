FROM ubuntu:20.04

LABEL name="ML Template docker image" \
  maintainer="ml_developer@greystonevn.com" \
  description="ML Template Project"

ARG USER=greystone
ARG UIDRH=1000
ARG GIDRH=1000

ENV \
  APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1 \
  DEBIAN_FRONTEND=noninteractive \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8

USER root
# install prerequisites to run qtcreator, tools and Qt
RUN \
  apt-get update --quiet \
  && apt-get install --yes --quiet --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    gnupg \
    wget \
    curl \
    sudo zip unzip locate \
    git \
    vim \
    patch \
    ssh \
    make \
    p7zip-full \
    xterm \
    xdg-utils \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-xfixes0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libgssapi-krb5-2 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libharfbuzz-icu0 \
    libegl1-mesa-dev \
    libglu1-mesa-dev  \
    openssh-client \
    locales \
    libsm6 \
    libice6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libdbus-1-3 \
    libxi6 \
    libxcb-shape0 \
    libxcb-randr0 \
    zbar-tools \
  && apt-get --yes autoremove \
  && apt-get clean autoclean \
  && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*


# add user for development
RUN \
  mkdir -p /home/${USER} \
  && groupadd -g ${GIDRH} ${USER} \
  && useradd -d /home/${USER} -s /bin/bash -m ${USER} -u ${UIDRH} -g ${GIDRH} \
  && echo "${USER} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USER} \
  && chmod 0440 /etc/sudoers.d/${USER} \
  && chown ${UIDRH}:${GIDRH} -R /home/${USER}

RUN mkdir -p /opt/conda && chown -R ${USER}:${USER} /opt/conda
ENV PATH /opt/conda/bin:$PATH
WORKDIR /opt/conda
USER ${USER}

# Leave these args here to better use the Docker build cache
ARG CONDA_VERSION=py38_22.11.1-1

RUN set -x && \
    UNAME_M="$(uname -m)" && \
    if [ "${UNAME_M}" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh"; \
        SHA256SUM="473e5ecc8e078e9ef89355fbca21f8eefa5f9081544befca99867c7beac3150d"; \
    fi && \
    wget "${MINICONDA_URL}" -O miniconda.sh -q && \
    echo "${SHA256SUM} miniconda.sh" > shasum && \
    if [ "${CONDA_VERSION}" != "latest" ]; then sha256sum --check --status shasum; fi && \
    mkdir -p /opt && \
    bash miniconda.sh -b -u -p /opt/conda && \
    rm miniconda.sh shasum && \
    sudo ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

RUN python --version
RUN which python
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
RUN rm requirements.txt

WORKDIR /home/${USER}

CMD ["/bin/bash"]