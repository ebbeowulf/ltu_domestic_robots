FROM ros:humble
ARG USERNAME=emartinso
ARG USER_UID=1001
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
RUN apt-get install -y python3-pip ros-humble-cv-bridge ros-humble-vision-msgs
RUN apt-get install -y ros-humble-image-transport
COPY requirements.txt .
RUN pip install -r requirements.txt
ENV SHELL=/bin/bash

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************
RUN mkdir /ros2_ws
RUN chown -R $USERNAME:$USERNAME /ros2_ws
WORKDIR /ros2_ws
ENV ROS_DISTRO=humble
ENV ROS_DOMAIN_ID=32
ENV ROS_LOCALHOST_ONLY=0
ENV ROS_PYTHON_VERSION=3

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
CMD ["/bin/bash"]
