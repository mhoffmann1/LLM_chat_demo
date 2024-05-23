# LLM AI Chat demo

## What is this?

Jupyter notebook Python code representing basic deployment/functionality of AI chat using transformers, langchain and chromadb Python libraries

## Requirements/Setup

### HW requirements

* 12 GB RAM
* As many cpu cores as possible
* optional: NVIDIA GPU

### Software

* Python 3.10
* Python libraries listed in `requirements.txt` file

## How to run this code?

The code is intended to be run as an interactive Jupyter Notebook. There are several options to use it:

* [Recommended] Build the included Dockerfile and run Jupyter Notebook from it
* Load the Jupyter Notebook .ipynb file using Visual Studio Code, you will need to install Python interpreter and the dependencies from requirements.txt file manually
* Run without Jupyter Notebook, just use the basic LLM_chat.py (exported from the .ipynb file) after intalling the packages from requirements.txt file

### Steps to run using Jupyter Notebeok from docker container

1. Build the docker image (run from directory where the Dockerfile is located)

    ```bash
    docker build --network host --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -t jupyter-app:1.0 .
    ```

1. Prepare env file. It should include the Huggging Face access token and proxies (if you need them). See `env_file` to see how the file should look like

1. Run the container

    ```bash
    sudo docker run -d -p 5000:8888 --env-file {env_file} --name jupyter-app jupyter-app:1.0 jupyter-notebook --no-browser --config jupyter_notebook_config.py
    ```

1. Find the access token for Jupyter Notebook in the docker logs

    ```bash
    sudo docker logs jupyter-app
    ```

    Look for lines similar to this, copy the value of the token:

    ```bash
    [I 2024-05-23 15:39:39.543 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
    [C 2024-05-23 15:39:39.545 ServerApp]

    To access the server, open this file in a browser:
        file:///home/user/.local/share/jupyter/runtime/jpserver-1-open.html
    Or copy and paste one of these URLs:
        http://c82fa4158f44:8888/tree?token=a9b80d1f6d2719853831a4f7ca4d246faa64074b598f4a5c
        http://127.0.0.1:8888/tree?token=a9b80d1f6d2719853831a4f7ca4d246faa64074b598f4a5c
    ```

1. Use `http://localhost:5000/` in your browser to access the Jupyter Notebook main page. Use the token value from previous point to gain access.

1. Upload all files using Jupyter GUI, the files required to run the demo are: LLM_Chat.ipynb, llama-2-7b-chat.Q4_K_M.gguf and game_rulebook.pdf

    >Optional: You can run the docker container and mount a directory from your local filesystem (/srv/jupyter_resources in this example) to /home/user/jupyter-notebook with all the required files:

    ```bash
    sudo docker run -d -p 5000:8888 --env-file {env_file} -v /srv/jupyter_resources:/home/user/jupyter-notebook:rw --name jupyter-app jupyter-app:1.0 jupyter-notebook --no-browser --config jupyter_notebook_config.py
    ```

1. Open the LLM_Chat.ipynb via Jupyter Notebook by double clicking it

## Useful links

* [IBM: Introduction to Generative AI](https://skills.yourlearning.ibm.com/activity/MDL-388)
* [LLM introduction](https://blog.dataiku.com/large-language-model-chatgpt)
* [Generating text - greedy search, beam search etc.](https://huggingface.co/blog/how-to-generate)
* [How LLMs know when to end?](https://pub.towardsai.net/how-llms-know-when-to-stop-generating-b82a9a57e2c4)
* [Transformers documentation](https://huggingface.co/docs/transformers/index)
* [Customize BufferMemory](https://python.langchain.com/docs/modules/memory/conversational_customization)
* [Chat templates](https://huggingface.co/docs/transformers/main/chat_templating)
* [Stopping condition](https://discuss.huggingface.co/t/implimentation-of-stopping-criteria-list/20040/13)
* [Buffer Memory types](https://medium.com/@michael.j.hamilton/conversational-memory-with-langchain-82c25e23ec60) - sometimes available for free, otherwise paywall
* [Leonardo DiCaprio calculator example](https://python.langchain.com/docs/integrations/chat/huggingface/)
