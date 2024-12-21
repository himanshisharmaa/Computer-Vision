# Deployment

Deployment is the process of releasing and configuring software, applications, or systems into a production environment or a specific target environment where end users can access and use it. This involves activities such as installation, configuration, testing and performance  monitoring to ensure that the system operates correctly in its intended environment.

## Types of Deployment
1. Cloud Deployment 
- The computer vision model is hosted on cloud services like AWS, Azure or Google Cloud  and processes images/videos sent from client devices.
- Applications with high processing needs, scalable requirements, or where real-time processing is not critical.

- Artifacts
    - Trained model file (e.g., .h5, .onnx, .pt, .tflite).
    - REST API interface or deployment scripts.
    - Cloud infrastructure setup scripts (e.g., Terraform, CloudFormation).

- Frameworks
    - TensorFlow Serving, PyTorch Serve.
    - AWS Sagemaker, Google AI Platform, Azure Machine Learning.
    - Kubernetes (for container orchestration).
    - Docker (for containerized deployment).

- Advantages: 
    - High scalability
    - Easier Integration with other services.
    - Centralized management.

- Disadvantages:
    - Latency due to data transfer.
    - Dependency on internet connectivity.
    - Privacy concerns with sensitive data.

2. Edge Deployment
- The computer vision model is deployed directly on edge devices like cameras, smartphones or IoT devices.
- Applications needing real-time inference or operating in environments with limited or no internet connectivity.

- Artifacts
    - Optimized model for edge devices (e.g., .tflite, .onnx, .pt).
    - Preloaded libraries for edge hardware (e.g., TensorRT, OpenVINO).
    - Device-specific setup/configuration files.

-  Frameworks
    - TensorFlow Lite, ONNX Runtime.
    - NVIDIA TensorRT, Intel OpenVINO.
    - Edge platforms: AWS IoT Greengrass, Azure IoT Edge, Google Edge TPU.


- Advantages:
    - Low Latency
    - Offline operation.
    - Enhanced privacy as data stays on the device.
- Disadvantages:
    - Limited computational power
    - May require model optimization for deployment.

3. On-Premise Deployment
- The model is hosted on local servers or hardware within an organization's infrastructure.
- Applications requiring high data privacy, low-latency processing or regulatory compliance.

- Artifacts:
    - Model in deployable format (e.g., .h5, .onnx, .pb).
    - Docker containers for server setup.
    - Infrastructure automation scripts.

- Frameworks
    - TensorFlow Serving, PyTorch Serve.
    - NVIDIA Triton Inference Server.
    - Apache Kafka (for stream processing).
    - OpenCV (for preprocessing and inference).

- Advantages:
    - Full control over infrastructure
    - Enhanced data security
    - Low Latency for local networks

- Disadvantages:
    - High setup and maintenance costs.
    - Limited scalability compared to the cloud.

4. Hybrid Deployment
- Combines cloud and edge/on-premise deployment. The cloud handles heavy computations while edge devices perform lightweight processing or data collection.
- Applications balancing real-time processing and scalable storage/analytics.

- Artifacts
    - Cloud-based API for heavy processing.
    - Lightweight edge models for real-time inference.
    - Communication pipelines between edge and cloud.

- Frameworks
    - Cloud: TensorFlow Serving, PyTorch Serve.
    - Edge: TensorFlow Lite, NVIDIA TensorRT.
    - Middleware: MQTT, gRPC, or REST APIs.

- Advantages:
    - Flexible resource allocation.
    - Combines strengths of cloud and edge solutions.

- Disadvantages:
    - Increased complexity in system design.
    - Requires robust connectivity between cloud and edge.

5. Web Deployment
-  The model is integrated into web applications using tools like TensorFlow.js or WebAssembly.
- Browser-based applications for image classification, object detection, or face recognition.

- Artifacts
    - Model converted for browser compatibility (e.g., TensorFlow.js format).
    - JavaScript code for inference and integration.

- Frameworks
    - TensorFlow.js.
    - ONNX.js.
    - WebAssembly (for model execution in the browser).
    - WebRTC (for real-time streaming).

- Advantages:
    - Platform-independent access (runs in the browser).
    - No installation required for users.

- Disadvantages:
    - Limited processing power of browsers.
    - May require internet access for additional resources.

6. Batch Processing Deployment
- The model is deployed to process large volumes of pre-recorded or historical image/video data.
- Analysis of surveillance footage or large datasets for insights.

- Artifacts
    - Scripts to process large datasets.
    - Scheduler setup for batch jobs.
    - Model and preprocessing pipelines.

- Frameworks
    - Apache Spark, Hadoop (for large-scale processing).
    - TensorFlow, PyTorch.
    - Kubernetes (for managing batch jobs).
    - Message Queues: RabbitMQ, Kafka.

- Advantages:
    - Efficient for large-scale processing.
    - Can be scheduled for low-usage periods to optimize resources.

-Disadvantages:
- Not suitable for real-time applications.