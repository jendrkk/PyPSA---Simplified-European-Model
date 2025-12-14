"""
SSH helpers for transfer and remote execution.
Password-based SSH supported via interactive prompt.
"""
from __future__ import annotations
import getpass
import os
import shlex
import subprocess
from dataclasses import dataclass

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
    return getpass.getpass(prompt="Enter SSH password (will be used by ssh/scp): ")


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
