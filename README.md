# CheckThat! Track Implementation

Little implementation for the CheckThat! tasks. This code base will continue to evolve and things will change (in unpredicable ways). The current layout should be easy enough to work with. I've used Docker to take away most of the pain.

Right now it just implements Task 2 using a cloud LLM API provided by [together.ai](https://together.ai/) and very conventional Python libraries for templating and natural language/data manipulation. I'm running this on latest Python, this will probably run on versions as old as 3.12 -- not sure about pre-3.12.

## Running the Implementation

Use these commands in the workspace.
You don't have to use Docker. Podman will also work.

```bash
export CHECKTHAT_CACHE_DIR=$(pwd)/.checkthat_cache
export TRANSFORMERS_CACHE_DIR=$(pwd)/.transformers-cache
mkdir -p "$CHECKTHAT_CACHE_DIR"
mkdir -p "$TRANSFORMERS_CACHE_DIR"
docker build -t checkthat_track:latest .
docker run --rm -e "TOGETHER_API_TOKEN=<your token here>" \
    --mount "type=bind,src=$CHECKTHAT_CACHE_DIR,dst=/usr/local/cs881/.checkthat_cache" \
    --mount "type=bind,src=$TRANSFORMERS_CACHE_DIR,dst=/home/${USERNAME:-ml-user}/.cache/torch" \
    checkthat_track:latest make run-all
```

You can change the name of the cache on your host system if you want.
The volume source dir outside the container can be changed.

If you want, you can bake more information into the docker image by setting build arguments.
There is one for the API key and various directories used. Look at the Dockerfile for more information.

## Sample Data

Sample eval tables are included so you don't have to run the LLM over everything.
To just calculate the METEOR score from the sample data, run (in the container): `./task2tool checkthat-task2 -nq together-ai <profile>`

Be sure to specify the profile you want to calculate the METEOR score for.
