#!/bin/bash
echo greystone | sudo -S docker run -id --gpus device=all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/greystone/StorageWall/model_template/Model_Triton:/models --name triton_server nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/models --strict-model-config=false --model-control-mode=explicit