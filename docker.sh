#!/bin/bash
# docker build -t rainy/unet:0.0.1 -f ./Dockerfile .
docker stop RETINA_UNET && docker rm RETINA_UNET
docker run -itd --name RETINA_UNET --rm \
    --gpus all \
    --net host \
    -v "$PWD":/workspace/RETINA_UNET \
    rainy/unet:0.0.1

docker exec -it RETINA_UNET /bin/bash /workspace/RETINA_UNET/run.sh
