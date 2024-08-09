option=$1

if [ $option = up ]; then
    # Start container
    docker compose -f docker-compose.yaml up -d
elif [ $option = force ]; then
    # Rebuild the docker image and start container
    docker compose -f docker-compose.yaml up -d --force-recreate
elif [ $option = down ]; then
    # Stop container
    docker compose -f docker-compose.yaml down -v
elif [ $option = rm ]; then
    # Stop the container and delete the image
    docker compose -f docker-compose.yaml down -v --rmi all
else
    echo "Command is wrong"
fi
