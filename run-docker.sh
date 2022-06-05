CONTAINER_NAME=eage2022_hackathon
SUBDIR_NAME=project
TAG=latest
 
WORKDIR=$PWD
 
echo "Starting $CONTAINER_NAME:$TAG container..."
 
docker build -t $CONTAINER_NAME .
 
# --rm \
sudo docker run --runtime=nvidia \
    -it \
    --shm-size 16G \
    --gpus all \
    --privileged \
    -v /dev:/dev \
    -v $WORKDIR:/workspace/$SUBDIR_NAME \
    --name $SUBDIR_NAME \
    $CONTAINER_NAME:$TAG