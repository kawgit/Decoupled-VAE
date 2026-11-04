"""Where a run's files go: artifacts/<job>/ under the working directory (the repo, on the Mac and on SCC). The job
is named by $DVAE_JOB (SCC's holders set it to the experiment's name, which scc/gpus.sh shows), else by the
script and the time it started."""

import functools
import os
import sys
import time

ROOT = "artifacts"


@functools.cache
def job_dir():
    """This run's folder, created on first use."""
    script = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "run"
    name = os.environ.get("DVAE_JOB") or f"{script}-{time.strftime('%Y%m%d-%H%M%S')}"
    path = os.path.join(ROOT, name)
    os.makedirs(path, exist_ok=True)
    return path


def artifact(filename):
    """A file in this run's folder."""
    return os.path.join(job_dir(), filename)
