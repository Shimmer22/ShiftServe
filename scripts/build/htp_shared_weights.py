#!/usr/bin/env python3
"""在 wins 的只读运行时上构建独立的 CPU/HTP 共享权重实验包。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4


BASE = "/home/wins/ai-workspace/tools/llama.cpp-hexagon-build"
PROBES = "/home/wins/ai-workspace/tools/llama.cpp-hexagon-probes"
IMAGE = "ghcr.io/snapdragon-toolchain/arm64-android:v0.3"


def compile_overlay(git_commit: str = "unknown", build_id: str = "unknown", extra_flags: list[str] | None = None):
    base = Path("/base/build-snapdragon")
    probe = Path("/probe")
    native = probe / "native"
    source = Path("/base/src")
    hexagon = base / "bin/libggml-hexagon.so"
    (probe / "hexagon-sha256.txt").write_text(hashlib.sha256(hexagon.read_bytes()).hexdigest() + "\n")
    entries = json.loads((base / "compile_commands.json").read_text())

    def rewrite(arg):
        return arg.replace("/workspace/src", str(source)).replace("/workspace/build-snapdragon", str(base))

    relative_paths = (
        "ggml/src/ggml-cpu/ggml-cpu.cpp", "ggml/src/ggml-cpu/traits.cpp",
        "src/llama-context.cpp", "src/llama-context.h",
    )
    overlay = probe / "overlay"
    for relative in relative_paths:
        dest = overlay / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((source / relative).read_bytes())
    for name in ("htp_shared_weights_runtime.patch", "phone-eval-runtime.patch"):
        patch = native / "backend" / name
        subprocess.run(["git", "apply", "--whitespace=error", str(patch)], cwd=overlay, check=True)

    def build_object(relative):
        original = source / relative
        patched = overlay / relative
        entry = next(item for item in entries if item["file"] == "/workspace/src/" + relative)
        args = [rewrite(arg) for arg in shlex.split(entry["command"])]
        index = args.index("-o") + 1
        old_object = args[index]
        new_object = str(patched.with_suffix(".o"))
        args[index] = new_object
        args[args.index(str(original))] = str(patched)
        args += ["-I" + str(original.parent), "-I/probe", "-I/probe/native/backend"]
        subprocess.run(args, cwd=base, check=True)
        return old_object, new_object

    def relink(name, objects):
        target = f"bin/{name}"
        trace = subprocess.check_output(["ninja", "-C", str(base), "-t", "commands", target], text=True)
        args = next(shlex.split(line) for line in trace.splitlines() if f"-o {target} " in line and "-shared" in line)
        args = args[args.index("&&") + 1:] if args[0] == ":" else args
        args = args[:args.index("&&")] if "&&" in args else args
        args = [objects.get(arg, rewrite(arg)) for arg in args]
        args[args.index("-o") + 1] = str(probe / name)
        subprocess.run(args, cwd=base, check=True)

    relink("libggml-cpu.so", dict(build_object(path) for path in relative_paths[:2]))
    relink("libllama.so", dict([build_object("src/llama-context.cpp")]))

    compiler = "/opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++"
    subprocess.run([
        compiler, "-std=c++17", "-O2", "-fPIE", "-pie",
        f'-DA2O_GIT_COMMIT="{git_commit}"', f'-DA2O_BUILD_ID="{build_id}"',
        *(extra_flags or []),
        "-I/base/src/include", "-I/base/src/ggml/include", "-I/base/src/tools/mtmd",
        "-I" + str(native), "-I/base/src/common", "-I/base/src/vendor", "-I/base/src/vendor/cpp-httplib",
        str(native / "session/phone_service.cpp"), "/base/src/vendor/cpp-httplib/httplib.cpp",
        "-L/probe", "-L/base/build-snapdragon/bin", "-lllama", "-lllama-common", "-lmtmd",
        "-lggml", "-lggml-base", "-llog", "-ldl", "-lm", "-o", str(probe / "agent"),
    ], check=True)


def main():
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--build-id", metavar="LABEL", help="构建标签，最终 ID 自动追加唯一后缀")
    p.add_argument("--prefill-weight-ab", action="store_true", help="启用诊断端点 /a2o/prefill-weight-ab")
    args = p.parse_args()
    output = args.output.resolve()
    output.relative_to(root / "results")
    label = args.build_id or output.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
        p.error("构建标签仅允许字母、数字、下划线、点和连字符")
    output.mkdir(parents=True, exist_ok=False)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "src/a2o/native", str(Path(__file__).resolve())],
        cwd=root, text=True).strip())
    build_id = f"{label}-{uuid4().hex}"
    remote = PROBES + "/" + output.name
    subprocess.run(["ssh", "wins", shlex.join(["mkdir", "-m", "700", remote])], check=True)
    script = Path(__file__).resolve()
    with TemporaryDirectory(prefix="a2o-build-") as staging:
        staged = Path(staging)
        native = shutil.copytree(root / "src/a2o/native", staged / "native")
        build_script = Path(shutil.copy2(script, staged))
        sources = {
            str(Path("src/a2o/native") / path.relative_to(native)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(native.rglob("*")) if path.is_file()
        }
        sources[str(script.relative_to(root))] = hashlib.sha256(build_script.read_bytes()).hexdigest()
        sources_digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        subprocess.run(["scp", "-r", str(native), str(build_script), f"wins:{remote}/"], check=True)
    extra_flags = ["-DA2O_PREFILL_WEIGHT_AB"] if args.prefill_weight_ab else []
    command = [
        "docker", "run", "--rm", "--pull=never", "-v", BASE + ":/base:ro", "-v", remote + ":/probe",
        IMAGE, "python3", "/probe/htp_shared_weights.py", "--inside", git_commit, build_id, *extra_flags,
    ]
    with (output / "build.log").open("w") as log:
        subprocess.run(["ssh", "wins", shlex.join(command)], stdout=log, stderr=subprocess.STDOUT, check=True)
    names = ["agent", "libggml-cpu.so", "libllama.so"]
    subprocess.run(["scp", *(f"wins:{remote}/{name}" for name in [*names, "hexagon-sha256.txt"]), str(output)], check=True)
    artifacts = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in names}
    (output / "manifest.json").write_text(json.dumps({
        "git_commit": git_commit, "build_id": build_id, "build_label": label, "dirty": dirty, "sources_sha256": sources_digest,
        "image": IMAGE, "remote_directory": remote, "artifacts": artifacts, "sources": sources,
        "hexagon_sha256": (output / "hexagon-sha256.txt").read_text().strip(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--inside":
        compile_overlay(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        main()
