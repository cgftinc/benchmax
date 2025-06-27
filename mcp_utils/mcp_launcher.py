"""
mcp_launcher.py – manage a whole fleet of Model-Context-Protocol servers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

USAGE
-----

# start everything described in mcp.json and block:
$ python mcp_launcher.py start

# same, but explicit config path:
$ python mcp_launcher.py start -c path/to/mcp.json

# list running servers (name, pid, uptime):
$ python mcp_launcher.py status

# stop a single server by name (or --pid):
$ python mcp_launcher.py stop weather

# graceful shutdown of *all* servers (SIGINT, then SIGTERM):
$ python mcp_launcher.py shutdown
"""
from __future__ import annotations

import argparse, json, os, signal, subprocess, time
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

################################################################################
# Helpers ----------------------------------------------------------------------

class MCPProcess:
    """Wrap a subprocess.Popen with metadata + friendly shutdown."""

    def __init__(self, name: str, entry: dict):
        self.name = name
        self.entry = entry
        self.started: Optional[datetime] = None
        self.proc: Optional[subprocess.Popen[str]] = None

    # ---------- lifecycle -----------------------------------------------------

    def start(self) -> None:
        """Spawn the process exactly as the host editor would."""
        cmd = [self.entry["command"], *self.entry.get("args", [])]
        env = {**os.environ, **self.entry.get("env", {})}
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        self.started = datetime.now()
        print(f"↑ [{self.name}] pid={self.pid}")

    def stop(self, *, timeout: float = 5.0) -> None:
        """Try SIGINT → wait → SIGTERM."""
        if not self.proc or self.proc.poll() is not None:
            return
        print(f"↓ [{self.name}] sending SIGINT …", end="", flush=True)
        with suppress(ProcessLookupError):
            os.kill(self.pid, signal.SIGINT)
        if self._wait(timeout):
            print("done.")
            return
        print("still running; sending SIGTERM …", end="", flush=True)
        with suppress(ProcessLookupError):
            os.kill(self.pid, signal.SIGTERM)
        self._wait(timeout)
        print("done.")

    def _wait(self, timeout: float) -> bool:
        try:
            if not self.proc:
                raise RuntimeError("process not started")
            self.proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    # ---------- convenience ---------------------------------------------------

    @property
    def pid(self) -> int:
        if not self.proc:
            raise RuntimeError("process not started")
        return self.proc.pid

    def uptime(self) -> str:
        if not self.started:
            return "-"
        delta = datetime.now() - self.started
        seconds = int(delta.total_seconds())
        mins, secs = divmod(seconds, 60)
        hrs, mins = divmod(mins, 60)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"

    def running(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)


class MCPFleet:
    """Start/stop many MCPProcess instances based on an mcp.json file."""

    def __init__(self, cfg_path: str | Path):
        self.cfg_path = Path(cfg_path)
        self.processes: Dict[str, MCPProcess] = self._load_config()

    # ---------- public API ----------------------------------------------------

    def start_all(self) -> None:
        for proc in self.processes.values():
            proc.start()
        print("⇢ All servers started.\n")

    def stop(self, name: Optional[str] = None, pid: Optional[int] = None) -> None:
        if name:
            self._by_name(name).stop()
        elif pid:
            self._by_pid(pid).stop()
        else:
            raise ValueError("specify --name or --pid")

    def shutdown(self) -> None:
        print("⇠ Shutting down fleet …")
        for proc in self.processes.values():
            proc.stop()
        print("✓ All servers stopped.")

    def status(self) -> None:
        print(f'{"NAME":<20}{"PID":>8}{"UPTIME":>10}')
        print("-" * 38)
        for p in self.processes.values():
            up = p.uptime() if p.running() else "dead"
            pid = p.pid if p.proc else "-"
            print(f"{p.name:<20}{pid:>8}{up:>10}")

    def wait_forever(self) -> None:
        """Block until SIGINT / SIGTERM, then shutdown."""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    # ---------- internals -----------------------------------------------------

    def _load_config(self) -> Dict[str, MCPProcess]:
        data = json.loads(self.cfg_path.read_text())
        servers = data.get("mcpServers") or data.get("mcp_servers")  # lenient
        if not servers:
            raise ValueError("no 'mcpServers' section found in config")
        return {name: MCPProcess(name, entry) for name, entry in servers.items()}

    def _by_name(self, name: str) -> MCPProcess:
        try:
            return self.processes[name]
        except KeyError:
            raise SystemExit(f"no server named '{name}'")

    def _by_pid(self, pid: int) -> MCPProcess:
        for p in self.processes.values():
            if p.pid == pid:
                return p
        raise SystemExit(f"no server with pid {pid}")

################################################################################
# CLI -------------------------------------------------------------------------

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Launch & control a fleet of MCP servers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-c",
        "--config",
        default="mcp.json",
        help="path to mcp.json (or .yaml) with an mcpServers section",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub_start = sub.add_parser("start", help="start all servers and block")
    sub_status = sub.add_parser("status", help="show running servers")
    sub_stop = sub.add_parser("stop", help="stop one server")
    sub_stop.add_argument("name", nargs="?", help="server name")
    sub_stop.add_argument("--pid", type=int, help="pid instead of name")
    sub_shutdown = sub.add_parser("shutdown", help="stop everything and exit")

    return p


def main(argv: List[str] | None = None) -> None:
    args = build_cli().parse_args(argv)
    fleet = MCPFleet(args.config)

    if args.cmd == "start":
        fleet.start_all()
        fleet.wait_forever()

    elif args.cmd == "status":
        fleet.status()

    elif args.cmd == "stop":
        fleet.stop(name=args.name, pid=args.pid)

    elif args.cmd == "shutdown":
        fleet.shutdown()


if __name__ == "__main__":
    main()