# Contributing to dyntapy

First, thank you for taking the time to contribute, we really appreciate it! :tada::+1:


## Submitting an issue

If you are running into problems running the code, see behaviour that defies expectation or just want to make some suggestions do [file an issue](https://gitlab.kuleuven.be/ITSCreaLab/public-toolboxes/dyntapy/-/issues/new).

If you are submitting a bug, we encourage you to provide a minimum reproducer. 
That is a script that allows anyone with a dyntapy installation to reproduce the error that you are experiencing.
This may not always be possible because some data files are required to run your code.

## Extending the codebase

There are detailed instructions for how to make a fork and submit merge requests [here](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html).
Follow the installation instructions in the [README.md](https://gitlab.kuleuven.be/ITSCreaLab/public-toolboxes/dyntapy) to create a new environment using the environment-dev.yml
with an editable version of dyntapy based on your forked repository via pip.

We use pre-commit hooks to ensure consistent formatting (Black) and compliance with PEP8 (Flake8). 

In the forked folder containing the .pre-commit file run
```shell
pre-commit install
```
to install the hooks. The hooks will now run on every commit that you make.
More information on pre-commit hooks can be found on their [website](https://pre-commit.com/).

Now you are ready to propose your changes!

Once you are happy with the changes you've made, verify if the tests pass
```shell
pytest
```
before submitting your merge request.

We will then review your code and merge as soon as possible.
