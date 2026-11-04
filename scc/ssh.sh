#!/bin/bash
# Open a shell on SCC in the repo there, with its Python module and venv active:
#   scc/ssh.sh              an interactive shell
#   scc/ssh.sh qstat -u kawgit56    run one command there instead
REPO=/projectnb/cs599bg/kawgit56/Decoupled-VAE
SETUP="cd $REPO && module load python3/3.13.8 && source ../venv/bin/activate"
if [ $# -eq 0 ]; then
  exec ssh -t scc1.bu.edu "$SETUP && exec bash -i"
fi
exec ssh scc1.bu.edu "$SETUP && $*"
