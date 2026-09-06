"""Linux code runner: isolated filesystem, unique uid, syscall allowlist, hard limits.

No model-supplied program runs until the fixed isolation preflight has passed.
Requires root only for setup/chroot; candidate execution always drops all uid/gid slots.
"""
from __future__ import annotations

import ctypes
import errno
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time


class SandboxUnavailable(RuntimeError):
    pass


ALLOW_SYSCALLS = """
read write readv writev close close_range lseek pread64 pwrite64
open openat stat lstat fstat newfstatat statx access faccessat faccessat2
readlink readlinkat getdents getdents64 getcwd chdir fchdir
mkdir mkdirat rmdir unlink unlinkat rename renameat renameat2
fcntl ioctl flock fsync fdatasync ftruncate truncate umask
mmap mmap2 mprotect munmap mremap madvise brk
rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack
futex futex_waitv set_tid_address set_robust_list rseq arch_prctl
clock_gettime clock_getres gettimeofday time nanosleep clock_nanosleep
getpid getppid gettid getuid geteuid getgid getegid getgroups
getrandom getrlimit ugetrlimit setrlimit prlimit64 getrusage times
sched_getaffinity sched_yield uname sysinfo restart_syscall exit exit_group
""".split()


def _seccomp(lib):
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(0x00050000 | errno.EPERM)  # default: reject with EPERM
    if not ctx:
        raise SandboxUnavailable("seccomp_init failed")
    try:
        for name in ALLOW_SYSCALLS:
            nr = lib.seccomp_syscall_resolve_name(name.encode())
            if nr >= 0 and lib.seccomp_rule_add(ctx, 0x7FFF0000, nr, 0) != 0:
                raise SandboxUnavailable("seccomp rule failed: " + name)
        if lib.seccomp_load(ctx) != 0:
            raise SandboxUnavailable("seccomp_load failed")
    finally:
        lib.seccomp_release(ctx)


def _child(root, uid, script, args):
    # Trusted setup completes before any candidate file is read or imported.
    import resource
    import runpy
    import pkgutil  # load runpy's dependency before replacing sys.path
    seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    libc = ctypes.CDLL(None, use_errno=True)
    os.chroot(root)
    os.chdir("/work")
    os.setgroups([])
    os.setresgid(uid, uid, uid)
    os.setresuid(uid, uid, uid)
    if os.getresuid() != (uid, uid, uid) or os.getresgid() != (uid, uid, uid):
        raise SandboxUnavailable("uid/gid drop failed")
    for limit, value in ((resource.RLIMIT_AS, 2 * 1024**3), (resource.RLIMIT_CPU, 11),
                         (resource.RLIMIT_FSIZE, 64 * 1024**2), (resource.RLIMIT_NOFILE, 64),
                         (resource.RLIMIT_NPROC, 1), (resource.RLIMIT_CORE, 0),
                         (resource.RLIMIT_STACK, 64 * 1024**2)):
        resource.setrlimit(limit, (value, value))
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise SandboxUnavailable("no_new_privs failed")
    os.environ.clear()
    os.environ.update({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "HOME": "/work", "TMPDIR": "/tmp"})
    stdlib = sysconfig.get_path("stdlib")
    sys.path[:] = [stdlib, str(Path(stdlib) / "lib-dynload"), "/work"]
    sys.dont_write_bytecode = True
    sys.argv[:] = [script] + args
    _seccomp(seccomp)
    runpy.run_path(script, run_name="__main__")


class Sandbox:
    def __init__(self):
        self.root = None
        self.report = None

    def prepare(self):
        if sys.platform != "linux" or not hasattr(os, "geteuid") or os.geteuid() != 0:
            raise SandboxUnavailable("Linux root setup is required for chroot; candidate runs unprivileged")
        try:
            ctypes.CDLL("libseccomp.so.2")
        except OSError as exc:
            raise SandboxUnavailable("Install libseccomp2 before running code evaluation") from exc
        self.root = Path(tempfile.mkdtemp(prefix="cap403_runtime_"))
        os.chmod(self.root, 0o755)
        stdlib = Path(sysconfig.get_path("stdlib"))
        if not str(stdlib).startswith("/usr/"):
            raise SandboxUnavailable("Use a system Python with its standard library under /usr")
        dest = self.root / str(stdlib).lstrip("/")
        shutil.copytree(stdlib, dest, ignore=shutil.ignore_patterns(
            "site-packages", "dist-packages", "__pycache__", "test", "tests", "idlelib", "tkinter", "ensurepip"))
        binaries = [Path(sys.executable).resolve()] + list((stdlib / "lib-dynload").glob("*.so"))
        dependencies = set()
        for binary in binaries:
            proc = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, timeout=15,
                                  env={"PATH": "/usr/bin:/bin", "LANG": "C"})
            for line in proc.stdout.splitlines():
                for token in line.split():
                    if token.startswith("/") and Path(token).is_file():
                        dependencies.add(Path(token))
        for source in dependencies:
            target = self.root / str(source).lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for path in self.root.rglob("*"):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        self.report = self.preflight()
        return self.report

    def execute(self, workdir, script, args, stdin=b"", timeout=10.0, capture_stdout=True):
        if self.root is None:
            raise SandboxUnavailable("Sandbox runtime has not been prepared")
        # Separate inode tree; only immutable runtime files are shared as hardlinks.
        case = Path(tempfile.mkdtemp(prefix="cap403_case_"))
        uid = 65536 + secrets.randbelow(2**30)
        os.chmod(case, 0o755)
        shutil.copytree(self.root, case, dirs_exist_ok=True, copy_function=os.link)
        work = case / "work"
        shutil.copytree(workdir, work)
        nonce_path = Path(workdir) / "_nonce.txt"
        expected_done = "_done_" + nonce_path.read_text(encoding="utf-8").strip() if nonce_path.is_file() else None
        (case / "tmp").mkdir()
        for path in [work, case / "tmp"] + list(work.rglob("*")):
            if path.is_symlink():
                raise SandboxUnavailable("Symlinks in input case are not permitted")
            os.chown(path, uid, uid)
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
        out_path, err_path = Path(workdir) / "stdout.txt", Path(workdir) / "stderr.txt"
        timed_out = False
        started = time.monotonic()
        try:
            with out_path.open("wb") as stdout, err_path.open("wb") as stderr:
                cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--child", str(case), str(uid), script] + list(args)
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=stdout if capture_stdout else subprocess.DEVNULL,
                                        stderr=stderr, close_fds=True, start_new_session=True,
                                        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
                try:
                    proc.communicate(stdin or b"", timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.communicate(timeout=5)
            done = expected_done is not None and (work / expected_done).exists()
            with out_path.open("rb") as output, err_path.open("rb") as errors:
                stdout_text = output.read(8 * 1024**2).decode("utf-8", "replace") if capture_stdout else ""
                errors.seek(0, os.SEEK_END)
                errors.seek(max(0, errors.tell() - 3000))
                stderr_text = errors.read().decode("utf-8", "replace")[-1500:]
            return {"rc": proc.returncode, "timed_out": timed_out,
                    "stdout": stdout_text, "stderr": stderr_text,
                    "done": done, "elapsed": round(time.monotonic() - started, 3)}
        finally:
            shutil.rmtree(case)

    def preflight(self):
        with tempfile.TemporaryDirectory(prefix="cap403_probe_") as work, tempfile.TemporaryDirectory(prefix="cap403_host_") as outside:
            canary = Path(outside) / "secret_canary"
            canary.write_text(secrets.token_hex(16))
            script = '''import errno, json, os, resource, socket
import hashlib, decimal, fractions, statistics, heapq, collections, math, random
def denied(fn, allowed=(errno.EPERM,)):
    try:
        fn()
    except OSError as e:
        return e.errno in allowed
    return False
checks = {
 "host_file_blocked": denied(lambda: open(CANARY).read(), (errno.ENOENT, errno.EACCES)),
 "network_blocked": denied(lambda: socket.socket()),
 "process_creation_blocked": denied(lambda: os.fork()),
 "exec_blocked": denied(lambda: os.execv('/work/probe.py', ['/work/probe.py'])),
 "host_signal_blocked": denied(lambda: os.kill(HOSTPID, 0)),
 "unprivileged": os.geteuid() >= 65536 and os.getegid() >= 65536,
 "no_host_paths": not any(os.path.exists(p) for p in ['/root', '/workspace', '/proc', '/dev', CANARY]),
 "clean_environment": set(os.environ) <= {'LANG', 'LC_ALL', 'HOME', 'TMPDIR'},
 "runtime_read_only": denied(lambda: open(os.__file__, 'wb'), (errno.EACCES,)),
 "process_limit": resource.getrlimit(resource.RLIMIT_NPROC) == (1, 1),
 "memory_limit": resource.getrlimit(resource.RLIMIT_AS) == (2147483648, 2147483648),
 "stdlib_works": math.isqrt(144) == 12 and str(fractions.Fraction(1, 3) * 3) == '1',
}
print(json.dumps(checks, sort_keys=True))
'''.replace("CANARY", repr(str(canary))).replace("HOSTPID", str(os.getpid()))
            Path(work, "probe.py").write_text(script, encoding="utf-8")
            result = self.execute(work, "probe.py", [], timeout=20)
            if result["rc"] != 0 or result["timed_out"]:
                raise SandboxUnavailable("Isolation preflight failed: " + result["stderr"][-2000:])
            try:
                checks = json.loads(result["stdout"])
            except ValueError as exc:
                raise SandboxUnavailable("Isolation preflight did not return JSON") from exc
            if not checks or not all(checks.values()):
                raise SandboxUnavailable("Isolation preflight checks failed: " + json.dumps(checks))
            return {"backend": "chroot_unique_uid_seccomp_allowlist", "passed": True, "checks": checks,
                    "python": sys.version, "syscalls_allowed": ALLOW_SYSCALLS,
                    "limits": {"memory_bytes": 2 * 1024**3, "cpu_seconds": 11, "file_bytes": 64 * 1024**2,
                               "open_files": 64, "processes": 1},
                    "namespace_isolation": False, "network_isolation": "all socket syscalls rejected"}

    def patch_code_module(self, module):
        if not self.report or not self.report["passed"]:
            raise SandboxUnavailable("Isolation preflight must pass before installing code scorer")
        def run_process(tmpdir, argv, stdin_bytes, timeout, capture_stdout):
            nonce = secrets.token_hex(8)
            Path(tmpdir, module.NONCE_FILE).write_text(nonce, encoding="utf-8")
            return self.execute(tmpdir, "_run.py", argv, stdin_bytes, timeout, capture_stdout)
        module._run_process = run_process

    def close(self):
        if self.root:
            shutil.rmtree(self.root, ignore_errors=True)
            self.root = None


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--child":
        _child(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5:])
    else:
        # No model endpoint is accessed by this fixed-code preflight.
        runner = Sandbox()
        try:
            print(json.dumps(runner.prepare(), indent=2))
        finally:
            runner.close()
