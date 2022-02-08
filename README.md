
# DataTriad NLP Project

[![Tests Status](https://github.com/datatriadian/stacksample/workflows/Testing/badge.svg?branch=main&event=push)](https://github.com/datatriadian/stacksample/actions?query=workflow%3ATesting+branch%3Amain+event%3Apush)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/datatriadian/stacksample/main.svg)](https://results.pre-commit.ci/latest/github/datatriadian/stacksample/main)

This repo is for the DataTriad NLP project. This is a work in progress.

## Downloading the data

Because of the large size of the data files they are not included in the repo and they need to be
downloaded. To do this create a `data` directory, the `.gitignore` file is set to ignore the `data`
directory so anything included here won't be added to git, then download and unzip the files from
[here](https://www.kaggle.com/stackoverflow/stacksample) into the newly created `data` directory.

## using the program

For help on what you can do run:

```sh
stacksample --help
```

For help on a specific section (for example training) run:

```sh
stacksample train --help
```
