# Vector database search
This app provides basic search of Wikipedia articles in vector Database.

## Requirements
- Git
- Docker [Docker website](https://www.docker.com/products/docker-desktop/)
- docker-compose [Docker Compose Installation](https://docs.docker.com/compose/install/)


## Installation

Clone the repo in your terminal app:

```git clone git@github.com:bitfroster/vector-db-search.git```

Change the project directory to app directory:

```cd vector-db-search```

Then you need to run the following command:

```docker-compose up```

You can start docker compose in detached mode to release terminal app/tab:
```docker-compose up -d``` [more info](https://docs.docker.com/engine/reference/commandline/compose_up/)

Installation could take a while and depends on your internet connection speed.

Afterwards you need to go to the ```app``` container and execute the commands:
```
docker-compose exec app bash
```
and then run:

```python main.py --download```

Afterwards you can use search query in cli:

```python main.py -q 'Some text to search'``` 

or 

```python main.py --query 'Some text to search'```

On the first run the script downloads the models, it could take some time.

Alternatively you can run command via docker-compose exec without logging to interactive shell:
```
docker-compose exec app python main.py -d
docker-compose exec app python main.py -q 'Some text to search'
```