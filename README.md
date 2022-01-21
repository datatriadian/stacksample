# DataTriad NLP Project

This repo is for the DataTriad NLP project. To use follow the stops below.

## Fork the project

In order to work on the project you will need your own fork. To do this click the "Fork" button on
this project.

Once the project is forked clone it to your local machine:

```sh
git clone git@github.com:your-user-name/stacksample.git
cd stacksample
git remote add upstream git@github.com:datatriadian/stacksample.git
```

This creates the directory stacksample and connects your repository to the upstream (main project) repository.

## Download the data 

Because of the large size of the data files they are not included in the repo and they need to be
downloaded. To do this create a `data` directory, the `.gitignore` file is set to ignore the `data`
directory so anything included here won't be added to git, then download the files from
[here](https://www.kaggle.com/stackoverflow/stacksample) into the newly created `data` directory.

## Working with the code

Note: This project uses Poetry to manage dependencies. If you do not already have Poetry installed you will need to install it with the instuctions [here](https://python-poetry.org/docs/master/#installation)

First the requirements need to be installed.

```sh
poetry install
```

## Creating a branch

You want your main branch to reflect only production-ready code, so create a feature branch for
making your changes. For example:

```sh
git checkout -b my-new-feature
```

This changes your working directory to the my-new-feature branch. Keep any changes in this branch
specific to one bug or feature so the purpose is clear. You can have many my-new-features and switch
in between them using the git checkout command.

When creating this branch, if you plan to keep your code in sync with the main project, make sure 
your main branch is up to date with the latest upstream main version. To update your local main branch,
you can do:

```sh
git checkout main
git pull upstream main --ff-only
```

## pre-commit
pre-commit hooks are included to run linting when you commit changes to you branch. To setup pre-commit for this project run:

```sh
pre-commit install
```

After this pre-commit will automatically run any time you check in code to your branches. You can also run pre-commit at any time with:

```sh
pre-commit run --all-files
```

## Testing

This project uses pytest for testing. In addition tox is available to run tests.

To run the tests run:

```sh
poetry run pytest
```

tox can be used to run both linting, and run the tests in all of the supported Python versions.
Note that you will need to have all the verions of Python installed for this to work.

```sh
poetry run tox
```

## Committing your code

Once you have made changes to the code on your branch you can see which files have changed by running:

```sh
git status
```

If new files were created that and are not tracked by git they can be added by running:

```sh
git add .
```

Now you can commit your changes in your local repository:

```sh
git commit -am 'Some short helpful message to describe your changes'
```

If you setup pre-commit and any of the tests fail the commit will be cancelled and you will need tox
fix any errors. Once the errors are fixed you can run the same git commit command again.

## Push your changes

Once your changes are ready and all linting/tests are passing you can push your changes to your forked repositry:

```sh
git push origin my-new-feature
```

origin is the default name of your remote repositry on GitHub. You can see all of your remote repositories by running:

```sh
git remote -v
```

## Making a Pull Request

If you would like to include your code into the main project you can make a pull request. To submit the pull request:

1. Navigate to your repository on GitHub
2. Click on the Pull Request button for your feature branch
3. You can then click on Commits and Files Changed to make sure everything looks okay one last time
4. Write a description of your changes in the Conversation tab
5. Click Send Pull Request

This request then goes to the repository maintainers, and they will review the code.

## Updating your pull request

Changes to your code may be needed based on the review of your pull request. If this is the case you
can make them in your branch, add a new commit to that branch, push it to GitHub, and the pull
request will be automatically updated. Pushing them to GitHub again is done by:

```sh
git push origin my-new-feature
```

This will automatically update your pull request with the latest code and restart the Continuous Integration tests.

Another reason you might need to update your pull request is to solve conflicts with changes that
have been merged into the main branch since you opened your pull request.

To do this, you need to rebase your branch:

```sh
git checkout my-new-feature
git fetch upstream
git rebase upstream/main
```

There may be some merge conficts that need to be resolved. After the feature branch has been updated
locally, you can now update your pull request by pushing to the branch on GitHub:

```sh
git push -f origin my-new-feature
```

## Delete your merged branch (optional)

Once your feature branch is accepted into upstream, you’ll probably want to get rid of the branch.
First, merge upstream main into your main branch so git knows it is safe to delete your branch:

```sh
git checkout main
git pull upstream main --ff-only
```

Then you can deleate the feature branch:

```sh
git branch -d my-new-feature
```

Make sure you use a lower-case -d, or else git won’t warn you if your feature branch has not actually been merged.

The branch will still exist on GitHub, so to delete it there do:

```sh
git push origin --delete my-new-feature
```
