"""Copy reviewed FPE text sources through the registered SSH route, without stdin streaming."""
import argparse
import base64
import hashlib
import json
import shlex
import subprocess
import zlib
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ssh-config", required=True)
    p.add_argument("--host", default="materials-dsw")
    p.add_argument("--control-path")
    args = p.parse_args()
    local = Path(__file__).resolve().parents[1]
    ssh = ["ssh", "-F", args.ssh_config]
    if args.control_path:
        ssh.extend(["-o", "ControlPath=" + args.control_path])
    config = subprocess.run([*ssh, "-G", args.host], stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, check=True).stdout
    if "stricthostkeychecking true" not in config or "batchmode yes" not in config:
        raise ValueError("registered SSH profile must enforce host keys and batch mode")
    if "hostname dsw-lnt0d9qe8zf1vqk6bi\n" not in config:
        raise ValueError("SSH profile resolves to another instance")
    files = [local / "pyproject.toml", local / "uv.lock", *sorted((local / "src").rglob("*.py")),
             *sorted((local / "deploy").glob("*.py")), *sorted((local / "deploy").glob("*.sh")), *sorted((local / "deploy/locks").glob("*.lock")),
             *sorted((local / "tests").glob("*.py"))]
    data = {str(f.relative_to(local)): f.read_text() for f in files}
    # Individual chunks remain below operating-system argument-size limits.
    groups = []
    group = {}
    for key, value in data.items():
        group[key] = value
        if len(json.dumps(group)) > 30000:
            groups.append(group)
            group = {}
    if group:
        groups.append(group)
    for group in groups:
        blob = base64.b64encode(zlib.compress(json.dumps(group).encode())).decode()
        script = ("import base64,zlib,json,pathlib,os\n"
                  "r=pathlib.Path('/mnt/workspace/fpe-self-consistency')\n"
                  "d=json.loads(zlib.decompress(base64.b64decode(" + repr(blob) + ")))\n"
                  "for k,v in d.items():\n"
                  " p=r/k; p.parent.mkdir(parents=True,exist_ok=True)\n"
                  " tmp=p.with_name(p.name+'.deploy-'+str(os.getpid()))\n"
                  " tmp.write_text(v); tmp.replace(p)\n"
                  "print('DEPLOYED',len(d))")
        result = subprocess.run([*ssh, args.host, "python3 -c " + shlex.quote(script)],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False)
        if result.returncode:
            raise RuntimeError("SSH deployment interrupted; inspect current files before continuing")
        print(result.stdout.strip())
    print(json.dumps({"source_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in data.items()}},
                     sort_keys=True))


if __name__ == "__main__":
    main()
