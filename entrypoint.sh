#!/bin/bash
test -f /.dockerenv || { echo "NOT IN DOCKER, aborting"; exit 1; }

git config --global credential.helper store
git config --global user.email "luke.e.hearon@gmail.com"
git config --global user.name "LukHearon"
echo "https://LukeHearon:$GIT_TOKEN@github.com" > ~/.git-credentials

exec claude --dangerously-skip-permissions
