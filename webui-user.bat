@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS= --api --cors-allow-origins http://localhost:5173 --precision autocast --deepdanbooru --xformers

call webui.bat
