
# ==========================================================
# Stage 1: Build OpenCV 4.10.0 and C++ YuNet Service
# ==========================================================
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Download OpenCV 4.10.0
WORKDIR /opt

RUN git clone --depth 1 --branch 4.10.0 \
    https://github.com/opencv/opencv.git opencv-src

# Compile only necessary OpenCV modules
RUN cmake -S /opt/opencv-src -B /opt/opencv-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/opencv \
    -DBUILD_LIST=core,imgproc,imgcodecs,dnn,objdetect \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_JAVA=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=OFF \
    -DWITH_IPP=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_TBB=OFF \
    -DWITH_FFMPEG=OFF \
    -DWITH_GSTREAMER=OFF \
    -DWITH_GTK=OFF \
    -DWITH_QT=OFF \
    -DWITH_OPENEXR=OFF \
    -DWITH_TIFF=OFF \
    -DWITH_WEBP=OFF \
    -DWITH_JPEG=ON \
    -DWITH_PNG=ON \
    -DWITH_PROTOBUF=ON \
    -DBUILD_PROTOBUF=ON \
    && cmake --build /opt/opencv-build --parallel 2 \
    && cmake --install /opt/opencv-build

# Build C++ service
WORKDIR /build

COPY CMakeLists.txt ./
COPY src/ ./src/
COPY include/ ./include/

RUN cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4 \
    && cmake --build build --parallel 2 \
    && strip build/yunet_service

# ==========================================================
# Stage 2: Runtime
# ==========================================================
FROM ubuntu:22.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive \
    PORT=8080 \
    HOST=0.0.0.0 \
    MODEL_PATH=/app/models/face_detection_yunet_2023mar.onnx \
    SCORE_THRESHOLD=0.6 \
    NMS_THRESHOLD=0.3 \
    LD_LIBRARY_PATH=/opt/opencv/lib

# Only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libjpeg-turbo8 \
    libpng16-16 \
    zlib1g \
    libstdc++6 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled OpenCV libraries
COPY --from=builder /opt/opencv /opt/opencv

WORKDIR /app

# Copy C++ executable
COPY --from=builder /build/build/yunet_service /app/yunet_service

# Copy model
COPY models/ /app/models/

# Verify runtime shared libraries
RUN ldd /app/yunet_service && \
    ! ldd /app/yunet_service | grep -q "not found"

# Non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s \
    --start-period=10s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8080}/health" || exit 1

ENTRYPOINT ["/app/yunet_service"]
