#!/bin/bash

# set up UV and get necessary packages
curl -LsSf https://astral.sh/uv/install.sh | sh
# source UV
source $HOME/.local/bin/env
uv sync

