# Lean MuJoCo + Python image (headless-friendly)
FROM python:3.11-slim

# System libs MuJoCo needs at runtime (no dev packages to keep it lean)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libgles2 \
    libglfw3 \
    libglew2.2 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxi6 \
    libxinerama1 \
    libxcursor1 \
    libglib2.0-0 \
    patchelf \
 && rm -rf /var/lib/apt/lists/*

# Headless rendering via EGL; keep Python tidy
ENV MUJOCO_GL=glx \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Python MuJoCo bindings (v3+)
RUN pip install --no-cache-dir mujoco
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    casadi \
    scipy \
    numpy \
    matplotlib \
    # cyipopt \
    # osqp \
    # cvxpy \
    scikit-learn
RUN apt-get update && apt-get install -y \
    libgl1 libgl1-mesa-dri libglu1-mesa \
    libegl1 libgles2 libglfw3 libosmesa6 \
    python3-opengl mesa-utils

# Optional: viewer & video I/O (uncomment if you need them)
RUN pip install --no-cache-dir mujoco-python-viewer imageio[ffmpeg]

# Non-root user & working dir
# RUN useradd -ms /bin/bash appuser
# USER appuser
WORKDIR /app

CMD ["/bin/bash"]
