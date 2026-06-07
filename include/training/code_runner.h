#pragma once
// ============================================================================
// Sandboxed code runner — fork+exec a Python subprocess with hard resource
// limits, optional network isolation, and parent-side wall-clock enforcement.
//
// Used by CodeContestsEnv (training/env.h) to execute model-generated code
// against test cases.  Designed to be safe-enough by default; for production
// use behind a real container/bubblewrap, point the runner at that instead.
//
// What's enforced:
//   - RLIMIT_AS         memory cap (default 256 MB)
//   - RLIMIT_CPU        CPU-time cap (default 5 s, SIGXCPU)
//   - RLIMIT_FSIZE      file-write cap (1 MB)
//   - RLIMIT_NOFILE     max open FDs (64)
//   - RLIMIT_NPROC      max child processes (64)
//   - PR_SET_PDEATHSIG  child dies if parent crashes
//   - PR_SET_NO_NEW_PRIVS  blocks setuid exploits
//   - setpgid(0,0)      isolates process group for group-kill on timeout
//   - parent wall_time  hard-killed via SIGKILL to the child's process group
//   - stdout/stderr     bounded byte limits (truncated, child not killed)
//   - unshare(CLONE_NEWUSER|CLONE_NEWNET)  best-effort network isolation
//     (silently degraded if unprivileged user namespaces are disabled)
//
// What's NOT enforced (caveats — document & swap in a real sandbox for prod):
//   - filesystem access  (child can read /etc, /proc, /home — not jailed)
//   - syscall surface    (no seccomp filter; relies on OS)
//   - kernel-side bugs
// ============================================================================

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sched.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>

struct ResourceLimits {
    int   wall_time_sec    = 5;        // parent-enforced hard wall clock
    int   cpu_time_sec     = 5;        // RLIMIT_CPU (soft kill via SIGXCPU)
    long  memory_mb        = 256;      // RLIMIT_AS (virtual memory)
    long  stdout_max_bytes = 64 * 1024;
    long  stderr_max_bytes = 16 * 1024;
    long  fsize_max_bytes  = 1 * 1024 * 1024;
    int   max_processes    = 64;
    int   max_open_files   = 64;
    bool  isolate_network  = true;
};

struct RunResult {
    enum class Status {
        Ok = 0,        // exited normally with code 0
        NonZeroExit,   // exited normally with non-zero code
        Timeout,       // wall-time exceeded → process group SIGKILL'd
        MemoryLimit,   // killed by SIGSEGV / SIGABRT (heuristic for ENOMEM)
        Signal,        // died on some other signal
        SpawnFailed    // fork/exec failed before the child ran
    };
    Status      status            = Status::Ok;
    int         exit_code         = 0;
    int         signal            = 0;
    double      wall_time_s       = 0.0;
    std::string stdout_str;
    std::string stderr_str;
    bool        stdout_truncated  = false;
    bool        stderr_truncated  = false;
    std::string spawn_error;       // populated on SpawnFailed
};

class CodeRunner {
public:
    virtual ~CodeRunner() = default;
    virtual RunResult run(const std::string& code,
                          const std::string& stdin_data,
                          const ResourceLimits& limits) = 0;
};

// ----------------------------------------------------------------------------
// SubprocessPythonRunner — fork() → setrlimit → unshare → exec python3 -c <code>
// ----------------------------------------------------------------------------
class SubprocessPythonRunner : public CodeRunner {
public:
    explicit SubprocessPythonRunner(std::string python_path = "python3")
        : python_(std::move(python_path)) {}

    RunResult run(const std::string& code,
                  const std::string& stdin_data,
                  const ResourceLimits& limits) override
    {
        RunResult res;

        int sin[2], sout[2], serr[2], errpipe[2];
        if (pipe2(sin,  O_CLOEXEC) || pipe2(sout, O_CLOEXEC) ||
            pipe2(serr, O_CLOEXEC) || pipe2(errpipe, O_CLOEXEC)) {
            res.status      = RunResult::Status::SpawnFailed;
            res.spawn_error = "pipe2 failed: " + std::string(strerror(errno));
            return res;
        }

        auto t0  = std::chrono::steady_clock::now();
        pid_t pid = fork();
        if (pid < 0) {
            for (int p : {sin[0], sin[1], sout[0], sout[1],
                          serr[0], serr[1], errpipe[0], errpipe[1]}) close(p);
            res.status      = RunResult::Status::SpawnFailed;
            res.spawn_error = "fork failed: " + std::string(strerror(errno));
            return res;
        }

        if (pid == 0) {
            // ─── child ───────────────────────────────────────────────────────
            // Don't go through the cleanup path on any error — _exit only.
            // Errors before exec are reported via errpipe[1] (FD_CLOEXEC so it
            // auto-closes on successful exec; parent reads EOF → exec worked).

            // Disable CLOEXEC on errpipe write end so we can write to it on errors.
            // (Actually we WANT CLOEXEC: on successful exec, parent sees EOF; on
            // pre-exec failure, we write before exec.)

            // Move pipe ends to fd 0/1/2.
            if (dup2(sin[0],  STDIN_FILENO)  < 0) child_fail(errpipe[1], "dup2 stdin");
            if (dup2(sout[1], STDOUT_FILENO) < 0) child_fail(errpipe[1], "dup2 stdout");
            if (dup2(serr[1], STDERR_FILENO) < 0) child_fail(errpipe[1], "dup2 stderr");

            // Close all pipe halves we no longer need (the dup2'd ones are now
            // 0/1/2; the originals were O_CLOEXEC so exec will close them, but
            // close defensively).
            close(sin[0]);  close(sin[1]);
            close(sout[0]); close(sout[1]);
            close(serr[0]); close(serr[1]);
            close(errpipe[0]);

            // Process-group isolation so the parent can SIGKILL the whole tree.
            if (setpgid(0, 0) < 0) child_fail(errpipe[1], "setpgid");

            // Die if parent dies.
            prctl(PR_SET_PDEATHSIG, SIGKILL);
            // Block setuid exploits and any kernel bug requiring elevated caps.
            prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);

            // Resource limits — soft = hard so the child can't relax them.
            auto setlim = [&](int what, rlim_t v, const char* tag) {
                rlimit r{v, v};
                if (setrlimit(what, &r) != 0) child_fail(errpipe[1], tag);
            };
            setlim(RLIMIT_AS,     (rlim_t)limits.memory_mb     * 1024 * 1024, "RLIMIT_AS");
            setlim(RLIMIT_CPU,    (rlim_t)limits.cpu_time_sec,                "RLIMIT_CPU");
            setlim(RLIMIT_FSIZE,  (rlim_t)limits.fsize_max_bytes,             "RLIMIT_FSIZE");
            setlim(RLIMIT_NOFILE, (rlim_t)limits.max_open_files,              "RLIMIT_NOFILE");
            setlim(RLIMIT_NPROC,  (rlim_t)limits.max_processes,               "RLIMIT_NPROC");

            // Network isolation — best effort.  Requires unprivileged user
            // namespaces (some hardened kernels disable them).
            if (limits.isolate_network) {
                if (unshare(CLONE_NEWUSER) == 0) {
                    if (unshare(CLONE_NEWNET) != 0) {
                        // Soft-fail: log and continue without network isolation.
                        dprintf(errpipe[1], "WARN unshare(NEWNET): %s\n", strerror(errno));
                    }
                } else {
                    dprintf(errpipe[1], "WARN unshare(NEWUSER): %s\n", strerror(errno));
                }
            }

            // exec python3 -c <code>
            // python's -I flag isolates from site-packages env vars; -B disables
            // .pyc writes (RLIMIT_FSIZE would block writes anyway).
            const char* argv[] = {
                python_.c_str(), "-I", "-B", "-c", code.c_str(), nullptr
            };
            execvp(python_.c_str(), const_cast<char* const*>(argv));
            // exec returned → failed.
            child_fail(errpipe[1], "execvp");
            _exit(127);   // unreachable
        }

        // ─── parent ──────────────────────────────────────────────────────────
        close(sin[0]); close(sout[1]); close(serr[1]); close(errpipe[1]);

        // Write stdin then close so child sees EOF.
        if (!stdin_data.empty()) {
            // Best effort — if child died, write fails with EPIPE.  Loop to
            // handle partial writes.
            const char* p = stdin_data.data();
            size_t left = stdin_data.size();
            while (left > 0) {
                ssize_t n = write(sin[1], p, left);
                if (n < 0) { if (errno == EINTR) continue; break; }
                p    += n;
                left -= (size_t)n;
            }
        }
        close(sin[1]);

        // Set non-blocking on stdout/stderr/errpipe so poll-drain can't stall.
        for (int fd : {sout[0], serr[0], errpipe[0]}) {
            int fl = fcntl(fd, F_GETFL, 0);
            fcntl(fd, F_SETFL, fl | O_NONBLOCK);
        }

        const auto wall_deadline = t0 +
            std::chrono::milliseconds((long long)limits.wall_time_sec * 1000);

        // Drain stdout/stderr/errpipe while reaping the child.  When the wall
        // deadline passes, SIGKILL the whole process group.
        struct pollfd pfds[3] = {
            { sout[0],   POLLIN, 0 },
            { serr[0],   POLLIN, 0 },
            { errpipe[0],POLLIN, 0 },
        };
        bool open0 = true, open1 = true, open2 = true;
        std::string errpipe_buf;
        bool   killed_for_timeout = false;
        bool   exited = false;
        int    status = 0;

        while (open0 || open1 || open2 || !exited) {
            auto now = std::chrono::steady_clock::now();
            int  timeout_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                                  wall_deadline - now).count();
            if (timeout_ms <= 0) {
                if (!killed_for_timeout) {
                    ::kill(-pid, SIGKILL);
                    killed_for_timeout = true;
                }
                timeout_ms = 50;   // brief wait for the kill to propagate
            }

            pfds[0].revents = pfds[1].revents = pfds[2].revents = 0;
            // Only poll FDs still open.
            pfds[0].fd = open0 ? sout[0]   : -1;
            pfds[1].fd = open1 ? serr[0]   : -1;
            pfds[2].fd = open2 ? errpipe[0]: -1;

            int pr = poll(pfds, 3, timeout_ms);
            if (pr < 0 && errno == EINTR) continue;

            auto drain = [&](int fd, std::string& buf, long cap, bool& truncated) -> bool {
                char tmp[4096];
                for (;;) {
                    ssize_t n = read(fd, tmp, sizeof(tmp));
                    if (n > 0) {
                        long remaining = cap - (long)buf.size();
                        if (remaining > 0) {
                            long take = std::min<long>(n, remaining);
                            buf.append(tmp, tmp + take);
                            if (take < n) truncated = true;
                        } else {
                            truncated = true;
                        }
                    } else if (n == 0) {
                        return false;        // EOF
                    } else {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) return true;
                        if (errno == EINTR) continue;
                        return false;
                    }
                }
            };

            if (open0 && (pfds[0].revents & (POLLIN | POLLHUP | POLLERR)))
                open0 = drain(sout[0], res.stdout_str, limits.stdout_max_bytes, res.stdout_truncated);
            if (open1 && (pfds[1].revents & (POLLIN | POLLHUP | POLLERR)))
                open1 = drain(serr[0], res.stderr_str, limits.stderr_max_bytes, res.stderr_truncated);
            if (open2 && (pfds[2].revents & (POLLIN | POLLHUP | POLLERR)))
                open2 = drain(errpipe[0], errpipe_buf, 4096, /*ignored*/res.stdout_truncated);

            // Non-blocking waitpid so we keep draining even after child exits.
            if (!exited) {
                pid_t w = waitpid(pid, &status, WNOHANG);
                if (w == pid) exited = true;
            }
        }
        // Reap if not yet (shouldn't happen because we waited above, but defensive).
        if (!exited) waitpid(pid, &status, 0);

        auto t1 = std::chrono::steady_clock::now();
        res.wall_time_s =
            std::chrono::duration<double>(t1 - t0).count();

        // Surface child-side errpipe diagnostics into stderr_str for visibility.
        // (Useful when sandbox setup fails partway.)
        if (!errpipe_buf.empty()) {
            if (!res.stderr_str.empty()) res.stderr_str += "\n";
            res.stderr_str += "[runner] " + errpipe_buf;
        }

        if (killed_for_timeout) {
            res.status = RunResult::Status::Timeout;
        } else if (WIFEXITED(status)) {
            res.exit_code = WEXITSTATUS(status);
            res.status    = (res.exit_code == 0) ? RunResult::Status::Ok
                                                 : RunResult::Status::NonZeroExit;
        } else if (WIFSIGNALED(status)) {
            res.signal = WTERMSIG(status);
            // SIGKILL is what we send on timeout — but if we didn't, treat it
            // as OOM (RLIMIT_AS hits SIGKILL on big allocations sometimes via
            // overcommit accounting).  SIGSEGV/SIGABRT also often mean OOM.
            if (res.signal == SIGSEGV || res.signal == SIGABRT || res.signal == SIGKILL)
                res.status = RunResult::Status::MemoryLimit;
            else if (res.signal == SIGXCPU)
                res.status = RunResult::Status::Timeout;
            else
                res.status = RunResult::Status::Signal;
        }

        close(sout[0]); close(serr[0]); close(errpipe[0]);
        return res;
    }

private:
    std::string python_;

    // Used inside the child to bail out — writes a short tag to the error pipe
    // so the parent can surface it.  Then _exit(127).
    static void child_fail(int err_fd, const char* tag) {
        dprintf(err_fd, "child: %s: %s\n", tag, strerror(errno));
        _exit(127);
    }
};
