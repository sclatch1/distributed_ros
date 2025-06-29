Project Setup: Distributed Task Scheduling System
This project uses Docker Compose to manage and run a distributed task scheduling architecture. It consists of two main components: core (coordinator node, task node and ros master) and robot (robot nodes).

Step 1: Build the Containers
In your terminal, run the following commands to build the required Docker images:

```docker
docker compose --profile core build
docker compose --profile robots build
```

Step 2: Run the Containers
Open two terminal windows.

In the first terminal, start the core services:

```docker
docker compose --profile core up
````

In the second terminal, start the robot service:

```docker
docker compose --profile robots up
````