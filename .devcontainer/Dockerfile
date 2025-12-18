FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel
ARG USERNAME=USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Delete user if it exists in container (e.g Ubuntu Noble: ubuntu)
RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
RUN apt-get update && apt-get upgrade -y
RUN apt-get install vim python3-pip cmake git -y

RUN apt-get install libgl1-mesa-glx libglx-mesa0 -y

RUN apt-get install \
    libopengl0 \
    libgl1 \
    libglx0 \
    libglvnd0 \
    libegl1 \
    libgles2 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libxcb1 \
    libxcb-xkb1 \
    libx11-6 \
    libx11-xcb1 \
    libfontconfig1 \
    libfreetype6 \
    libdbus-1-3 \
    libglib2.0-0 \
    libglib2.0-bin \
    libgtk-3-0 \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 -y

RUN apt-get install \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxcb1 \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libglib2.0-0 \
    libdbus-1-3 \
    libfontconfig1 \
    libfreetype6 \
    libopengl0 \
    x11-utils -y

# Install OpenCV dependencies
RUN apt-get install -y build-essential cmake pkg-config libjpeg-dev libtiff-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran python3-dev

RUN pip install --no-cache-dir --proxy=http://172.17.0.1:10808 \
    opencv-python \
    onnx \
    onnxruntime-gpu \
    onnx-simplifier

RUN pip install --no-cache-dir --proxy=http://172.17.0.1:10808 triton

RUN pip install --no-cache-dir --proxy=http://172.17.0.1:10808 \
    ipykernel \
    scipy \
    matplotlib

RUN pip install --no-cache-dir --proxy=http://172.17.0.1:10808 pycuda

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone
RUN apt-get update && apt-get install -y libopencv-dev
# # Install OpenCV from source
# RUN apt-get install -y wget unzip
# WORKDIR /tmp
# RUN wget https://github.com/opencv/opencv/archive/refs/tags/4.5.4.tar.gz && \
#     tar -xvzf 4.5.4.tar.gz && \
#     rm 4.5.4.tar.gz
# RUN cd /tmp/opencv-4.5.4 && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE \
#     -D CMAKE_INSTALL_PREFIX=/usr/local \
#     -D WITH_GTK=ON \
#     -D CMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
#     -D CMAKE_C_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
#     -D OPENCV_GENERATE_PKGCONFIG=ON .. && \
#     make -j8 && \
#     sudo make install && \
#     sudo ldconfig

# ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# RUN pip install --no-cache-dir \
#     tensorrt==10.7.0 \
#     --extra-index-url https://pypi.nvidia.com || \
#     pip install --no-cache-dir tensorrt

RUN pip install --no-cache-dir --proxy=http://172.17.0.1:10808 \
    imageio \
    onnxscript

ENV SHELL /bin/bash

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
CMD ["/bin/bash"]