"""
SSH helpers for transfer and remote execution.
Password-based SSH supported via interactive prompt.
"""
from __future__ import annotations
import getpass
import os
import shlex
import subprocess
import time
from dataclasses import dataclass

HOST_LIST = [f"gruenau{i}.informatik.hu-berlin.de" for i in range(3,9)]

@dataclass
class SSHConfig:
    host: str 
    user: str
    port: int = 22

    def target(self) -> str:
        return f"{self.user}@{self.host}"


def _ensure_password_prompt():
    """
    Ensure user is prompted for password in terminal.
    Returns the password (unused directly but forces TTY interaction).
    """
    return getpass.getpass(prompt="Enter SSH password: ")


def transfer_to_server(local_path: str, server_path: str, ssh_cfg: SSHConfig) -> None:
    """Transfer files to server via scp, prompting for password."""
    _ensure_password_prompt()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    cmd = [
        "scp",
        "-P", str(ssh_cfg.port),
        local_path,
        f"{ssh_cfg.target()}:{server_path}"
    ]
    subprocess.run(cmd, check=True)


def fetch_from_server(remote_path: str, local_path: str, ssh_cfg: SSHConfig) -> None:
    """Fetch files from server via scp, prompting for password."""
    _ensure_password_prompt()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    cmd = [
        "scp",
        "-P", str(ssh_cfg.port),
        f"{ssh_cfg.target()}:{remote_path}",
        local_path
    ]
    subprocess.run(cmd, check=True)


def run_remote_job(ssh_cfg: SSHConfig, remote_command: str, remote_workdir: str | None = None) -> int:
    """
    Run a command on the remote server via ssh. Returns exit code.
    If `remote_workdir` is provided, the command is executed within it.
    """
    _ensure_password_prompt()
    if remote_workdir:
        # Use POSIX-safe compound command
        wrapped = f"cd {shlex.quote(remote_workdir)} && {remote_command}"
    else:
        wrapped = remote_command
    cmd = [
        "ssh",
        "-p", str(ssh_cfg.port),
        ssh_cfg.target(),
        wrapped
    ]
    return subprocess.run(cmd).returncode


def queue_remote_job_slurm(
    ssh_cfg: SSHConfig,
    remote_command: str,
    remote_workdir: str | None = None,
    slurm_options: dict | None = None,
) -> int:
    """
    Queue `remote_command` on the remote host using Slurm's `sbatch`.

    slurm_options may contain keys: `job_name`, `time`, `nodes`, `ntasks`,
    `partition`, `mem`, `output`.

    Returns the exit code from the `ssh` invocation that submits the job.
    """
    _ensure_password_prompt()

    if remote_workdir:
        cmd_body = f"cd {shlex.quote(remote_workdir)} && {remote_command}"
    else:
        cmd_body = remote_command

    opts = slurm_options or {}
    job_name = opts.get("job_name", "pypsa_job")
    time_limit = opts.get("time", "01:00:00")
    nodes = opts.get("nodes", 1)
    ntasks = opts.get("ntasks", 1)
    partition = opts.get("partition")
    mem = opts.get("mem")
    output = opts.get("output", f"/tmp/{job_name}-%j.out")

    header_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks={ntasks}",
    ]
    if partition:
        header_lines.append(f"#SBATCH --partition={partition}")
    if mem:
        header_lines.append(f"#SBATCH --mem={mem}")

    header = "\n".join(header_lines)
    script = f"{header}\n\n{cmd_body}\n"

    unique = f"pypsa_job_{os.getpid()}_{int(time.time())}"
    remote_script = f"/tmp/{unique}.sh"

    # Use a single-quoted here-doc to avoid remote shell expansion of the script
    wrapped = (
        f"cat > {shlex.quote(remote_script)} <<'PYPSA_EOF'\n"
        f"{script}"
        f"\nPYPSA_EOF\n"
        f"sbatch {shlex.quote(remote_script)}"
    )

    cmd = [
        "ssh",
        "-p", str(ssh_cfg.port),
        ssh_cfg.target(),
        wrapped,
    ]

    return subprocess.run(cmd).returncode
