@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS= --api --precision autocast --deepdanbooru --xformers --skip-torch-cuda-test

call webui.bat
