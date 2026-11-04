#!/bin/bash
# Copy the repo's files (tracked or new, not ignored; uncommitted edits included) to the SCC copy.
cd "$(dirname "$0")/.." || exit 1
git ls-files -z --cached --others --exclude-standard | COPYFILE_DISABLE=1 tar --no-xattrs --null -czf - -T - |
  ssh scc1.bu.edu "tar --unlink-first -xzf - -C /projectnb/cs599bg/kawgit56/Decoupled-VAE"  # fresh files: running scripts keep theirs
