/*
 * kci_win_compat.h — POSIX → Windows shim for the kci_analyze server.
 * ===================================================================
 * The server (kci_analyze.c) is written against epoll + BSD sockets. This header
 * lets it build and run natively on Windows (mingw-w64), with no WSL and no Docker.
 *
 * What it provides on _WIN32:
 *   - Winsock headers + WSAStartup helper (kci_win_net_init)
 *   - epoll_* implemented over select() — safe here because the event loop is
 *     level-triggered-friendly (it drains reads to EWOULDBLOCK and only arms
 *     EPOLLOUT while a write is pending), so EPOLLET is simply ignored.
 *   - socket I/O wrappers (recv/send/closesocket) that translate WSA error codes
 *     into errno (EAGAIN/EINTR) so the existing errno checks keep working.
 *   - small stubs: accept4, fcntl-nonblock, fsync→_commit, SIGPIPE, TCP_QUICKACK.
 *
 * On non-Windows this header is empty; kci_analyze.c includes the real POSIX
 * headers and defines the KCI_SOCK_* macros to plain read/write/close.
 */
#ifndef KCI_WIN_COMPAT_H
#define KCI_WIN_COMPAT_H
#ifdef _WIN32

#ifndef FD_SETSIZE
#define FD_SETSIZE 1024            /* must precede winsock2.h; raises select() capacity */
#endif
/* Only the server TU (kci_analyze.c) needs sockets; it defines KCI_WIN_SOCKETS so
   that kci_vec.c / kci_fts.c don't pull in winsock (whose connect() would clash
   with a static function of the same name in kci_vec.c). */
#ifdef KCI_WIN_SOCKETS
#include <winsock2.h>              /* must come before windows.h */
#include <ws2tcpip.h>
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN        /* keep windows.h from pulling old winsock.h */
#endif
#include <windows.h>
#include <io.h>                    /* _commit, _get_osfhandle, _open/_read/_write/_close */
#include <fcntl.h>
#include <malloc.h>
#include <process.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdint.h>
#include <errno.h>
#include <signal.h>
#include <time.h>

#ifndef __MINGW32__
#ifndef _SSIZE_T_DEFINED
typedef SSIZE_T ssize_t;
#define _SSIZE_T_DEFINED
#endif
#endif

/* sysconf(_SC_NPROCESSORS_ONLN) -> Windows logical CPU count */
#ifndef _SC_NPROCESSORS_ONLN
#define _SC_NPROCESSORS_ONLN 1
#endif
static inline long sysconf(int name) {
    (void)name;
    SYSTEM_INFO si; GetSystemInfo(&si);
    return (long)si.dwNumberOfProcessors;
}

/* posix_memalign -> _aligned_malloc. NOTE: such memory must be released with
   _aligned_free, so the matching free site uses KCI_ALIGNED_FREE (below). */
static inline int posix_memalign(void **p, size_t align, size_t sz) {
    void *q = _aligned_malloc(sz, align);
    if (!q) return 12 /* ENOMEM */;
    *p = q;
    return 0;
}
#define KCI_ALIGNED_FREE(p)  _aligned_free(p)

/* ---- pthread subset used by the core ----
 *
 * CAVEAT (verified 2026-06): the _MSC_VER branch below is currently unreachable
 * with an unmodified tree. The supported Windows build is mingw-w64 (see
 * build_kci_analyze_win.bat), where __MINGW32__ is defined and real winpthreads
 * is used instead. A straight MSVC `cl` build does not compile anyway, because
 * the core uses GNU C extensions (e.g. __attribute__((packed)) on Slot in
 * kci_analyze.c). This shim is kept as a forward-looking stub so that, if the
 * core is ever made MSVC-clean, threading is already covered — but it has NOT
 * been exercised by a real MSVC compile. Do not assume MSVC support from its
 * presence. */
#ifdef _MSC_VER
typedef HANDLE pthread_t;
typedef CRITICAL_SECTION pthread_mutex_t;
typedef void pthread_attr_t;
typedef struct {
    void *(*fn)(void *);
    void *arg;
} kci_pthread_start_t;
static unsigned __stdcall kci_pthread_boot(void *opaque) {
    kci_pthread_start_t *st = (kci_pthread_start_t *)opaque;
    void *(*fn)(void *) = st->fn;
    void *arg = st->arg;
    free(st);
    fn(arg);
    return 0;
}
static inline int pthread_create(pthread_t *thr, const pthread_attr_t *attr,
                                 void *(*fn)(void *), void *arg) {
    (void)attr;
    kci_pthread_start_t *st = (kci_pthread_start_t *)malloc(sizeof(*st));
    uintptr_t h;
    if (!st) return ENOMEM;
    st->fn = fn;
    st->arg = arg;
    h = _beginthreadex(NULL, 0, kci_pthread_boot, st, 0, NULL);
    if (!h) {
        free(st);
        return errno ? errno : EAGAIN;
    }
    *thr = (HANDLE)h;
    return 0;
}
static inline int pthread_join(pthread_t thr, void **retval) {
    DWORD rc = WaitForSingleObject(thr, INFINITE);
    if (rc != WAIT_OBJECT_0) return EINVAL;
    if (retval) *retval = NULL;
    CloseHandle(thr);
    return 0;
}
static inline int pthread_mutex_init(pthread_mutex_t *mu, const void *attr) {
    (void)attr;
    InitializeCriticalSection(mu);
    return 0;
}
static inline int pthread_mutex_destroy(pthread_mutex_t *mu) {
    DeleteCriticalSection(mu);
    return 0;
}
static inline int pthread_mutex_lock(pthread_mutex_t *mu) {
    EnterCriticalSection(mu);
    return 0;
}
static inline int pthread_mutex_unlock(pthread_mutex_t *mu) {
    LeaveCriticalSection(mu);
    return 0;
}
#else
#include <pthread.h>
#endif

/* ---- file-I/O bits POSIX has but Windows spells differently ---- */
#define fsync(fd)   _commit(fd)
#ifndef O_NONBLOCK
#define O_NONBLOCK  0x4000         /* sockets only; never reaches a file open() */
#endif

#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
#ifdef _MSC_VER
static inline int clock_gettime(int clock_id, struct timespec *ts) {
    if (!ts) { errno = EINVAL; return -1; }
    if (clock_id == CLOCK_MONOTONIC) {
        static LARGE_INTEGER freq;
        static int freq_init = 0;
        LARGE_INTEGER ctr;
        if (!freq_init) {
            if (!QueryPerformanceFrequency(&freq)) { errno = EINVAL; return -1; }
            freq_init = 1;
        }
        QueryPerformanceCounter(&ctr);
        ts->tv_sec = (time_t)(ctr.QuadPart / freq.QuadPart);
        ts->tv_nsec = (long)(((ctr.QuadPart % freq.QuadPart) * 1000000000ll) / freq.QuadPart);
        return 0;
    }
    FILETIME ft;
    ULARGE_INTEGER ui;
    GetSystemTimeAsFileTime(&ft);
    ui.LowPart = ft.dwLowDateTime;
    ui.HighPart = ft.dwHighDateTime;
    ui.QuadPart -= 116444736000000000ull; /* FILETIME epoch -> Unix epoch */
    ts->tv_sec = (time_t)(ui.QuadPart / 10000000ull);
    ts->tv_nsec = (long)((ui.QuadPart % 10000000ull) * 100);
    return 0;
}
#endif

/* ---- minimal mmap() over Windows file mapping ----
 * Covers the read-only MAP_SHARED whole-file mapping the FTS disk mode uses.
 *
 * Signal capability to kci_fts.c: its `#define KCI_FTS_HAVE_MMAP 1` only lives in
 * the POSIX `#elif` branch, so without this the Windows build compiles the
 * `#ifndef KCI_FTS_HAVE_MMAP -> return NULL` stub of kci_fts_open_mmap and FTLOAD
 * MMAP / PACKLOAD MMAP always fail even though the mapping shim below is present. */
#ifndef KCI_FTS_HAVE_MMAP
#define KCI_FTS_HAVE_MMAP 1
#endif
#define PROT_READ   0x1
#define PROT_WRITE  0x2
#define MAP_SHARED  0x01
#define MAP_PRIVATE 0x02
#define MAP_FAILED  ((void *)-1)
#define MS_SYNC     4
static inline void *mmap(void *addr, size_t len, int prot, int flags, int fd, long off) {
    (void)addr; (void)flags;
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return MAP_FAILED;
    DWORD page = (prot & PROT_WRITE) ? PAGE_READWRITE : PAGE_READONLY;
    DWORD acc  = (prot & PROT_WRITE) ? FILE_MAP_WRITE : FILE_MAP_READ;
    HANDLE m = CreateFileMappingA(h, NULL, page, 0, 0, NULL);
    if (!m) return MAP_FAILED;
    void *p = MapViewOfFile(m, acc, (DWORD)((uint64_t)off >> 32),
                            (DWORD)((uint64_t)off & 0xFFFFFFFFu), len);
    CloseHandle(m);   /* the live view keeps the mapping alive */
    return p ? p : MAP_FAILED;
}
static inline int munmap(void *addr, size_t len) {
    (void)len;
    return UnmapViewOfFile(addr) ? 0 : -1;
}
static inline int msync(void *addr, size_t len, int flags) {
    (void)len; (void)flags;
    return FlushViewOfFile(addr, 0) ? 0 : -1;
}

/* ---- signals / socket options Windows lacks (kept as harmless stubs) ---- */
#ifndef SIGPIPE
#define SIGPIPE 13
#endif
#ifndef TCP_QUICKACK
#define TCP_QUICKACK 200           /* Linux-only; setsockopt fails and the caller ignores it */
#endif
#ifndef SOCK_NONBLOCK
#define SOCK_NONBLOCK 0
#endif
#ifndef SOCK_CLOEXEC
#define SOCK_CLOEXEC 0
#endif

#ifdef KCI_WIN_SOCKETS   /* ===== sockets + epoll: only the server TU needs these ===== */

/* ---- WSA error -> errno, so existing `errno == EAGAIN` checks work ---- */
static inline int kci_wsa_errno(void) {
    int e = WSAGetLastError();
    if (e == WSAEWOULDBLOCK) return EAGAIN;
    if (e == WSAEINTR)       return EINTR;
    return EIO;
}

/* ---- socket I/O wrappers (sockets are not CRT fds on Windows) ---- */
static inline long kci_sock_read(int fd, void *buf, unsigned long n) {
    int r = recv((SOCKET)fd, (char *)buf, (int)n, 0);
    if (r < 0) errno = kci_wsa_errno();
    return r;
}
static inline long kci_sock_write(int fd, const void *buf, unsigned long n) {
    int r = send((SOCKET)fd, (const char *)buf, (int)n, 0);
    if (r < 0) errno = kci_wsa_errno();
    return r;
}
static inline int kci_sock_close(int fd) { return closesocket((SOCKET)fd); }
#define KCI_SOCK_READ(fd, b, n)  kci_sock_read((fd), (b), (n))
#define KCI_SOCK_WRITE(fd, b, n) kci_sock_write((fd), (b), (n))
#define KCI_SOCK_CLOSE(fd)       kci_sock_close(fd)

/* ---- set a socket non-blocking (replaces fcntl(F_SETFL,O_NONBLOCK)) ---- */
static inline int kci_set_nonblock(int fd) {
    u_long m = 1;
    return ioctlsocket((SOCKET)fd, FIONBIO, &m);
}

/* ---- WSAStartup, called once from main ---- */
static inline void kci_win_net_init(void) {
    WSADATA w;
    WSAStartup(MAKEWORD(2, 2), &w);
}

/* ================= epoll over select() ================= */
#define EPOLLIN        0x001
#define EPOLLPRI       0x002
#define EPOLLOUT       0x004
#define EPOLLERR       0x008
#define EPOLLHUP       0x010
#define EPOLLET        0x80000000u   /* ignored: select() is level-triggered (loop drains) */
#define EPOLL_CTL_ADD  1
#define EPOLL_CTL_DEL  2
#define EPOLL_CTL_MOD  3
#define EPOLL_CLOEXEC  0

typedef union { void *ptr; int fd; uint32_t u32; uint64_t u64; } epoll_data_t;
struct epoll_event { uint32_t events; epoll_data_t data; };

#define KCI_EPOLL_MAX  FD_SETSIZE
typedef struct {
    int      used;
    int      nfds;
    SOCKET   fds[KCI_EPOLL_MAX];
    uint32_t ev [KCI_EPOLL_MAX];
    void    *ptr[KCI_EPOLL_MAX];
} kci_epoll_t;
static kci_epoll_t kci_epoll_pool[8];

static inline int epoll_create1(int flags) {
    (void)flags;
    for (int i = 0; i < 8; i++) {
        if (!kci_epoll_pool[i].used) {
            memset(&kci_epoll_pool[i], 0, sizeof(kci_epoll_t));
            kci_epoll_pool[i].used = 1;
            return i + 1;
        }
    }
    return -1;
}
static inline int epoll_create(int n) { (void)n; return epoll_create1(0); }
static inline int epoll_close(int epfd) {
    if (epfd >= 1 && epfd <= 8) kci_epoll_pool[epfd - 1].used = 0;
    return 0;
}

static inline int epoll_ctl(int epfd, int op, int fd, struct epoll_event *ev) {
    if (epfd < 1 || epfd > 8) { errno = EINVAL; return -1; }
    kci_epoll_t *ep = &kci_epoll_pool[epfd - 1];
    SOCKET s = (SOCKET)fd;
    int idx = -1;
    for (int i = 0; i < ep->nfds; i++) if (ep->fds[i] == s) { idx = i; break; }
    if (op == EPOLL_CTL_DEL) {
        if (idx < 0) return 0;
        ep->fds[idx] = ep->fds[ep->nfds - 1];
        ep->ev [idx] = ep->ev [ep->nfds - 1];
        ep->ptr[idx] = ep->ptr[ep->nfds - 1];
        ep->nfds--;
        return 0;
    }
    if (op == EPOLL_CTL_ADD) {
        if (idx >= 0) return 0;
        if (ep->nfds >= KCI_EPOLL_MAX) { errno = ENOSPC; return -1; }
        idx = ep->nfds++;
        ep->fds[idx] = s;
    }
    ep->ev [idx] = ev ? ev->events : 0;
    ep->ptr[idx] = ev ? ev->data.ptr : NULL;
    return 0;
}

static inline int epoll_wait(int epfd, struct epoll_event *out, int maxev, int timeout_ms) {
    if (epfd < 1 || epfd > 8) { errno = EINVAL; return -1; }
    kci_epoll_t *ep = &kci_epoll_pool[epfd - 1];
    fd_set rd, wr, ex;
    FD_ZERO(&rd); FD_ZERO(&wr); FD_ZERO(&ex);
    for (int i = 0; i < ep->nfds; i++) {
        if (ep->ev[i] & EPOLLIN)  FD_SET(ep->fds[i], &rd);
        if (ep->ev[i] & EPOLLOUT) FD_SET(ep->fds[i], &wr);
        FD_SET(ep->fds[i], &ex);
    }
    struct timeval tv, *ptv = NULL;
    if (timeout_ms >= 0) { tv.tv_sec = timeout_ms / 1000; tv.tv_usec = (timeout_ms % 1000) * 1000; ptv = &tv; }
    int r = (ep->nfds > 0) ? select(0, &rd, &wr, &ex, ptv)
                           : (Sleep(timeout_ms < 0 ? 0 : timeout_ms), 0);
    if (r < 0)  { errno = kci_wsa_errno(); return -1; }
    if (r == 0) return 0;
    int n = 0;
    for (int i = 0; i < ep->nfds && n < maxev; i++) {
        uint32_t e = 0;
        if (FD_ISSET(ep->fds[i], &rd)) e |= EPOLLIN;
        if (FD_ISSET(ep->fds[i], &wr)) e |= EPOLLOUT;
        if (FD_ISSET(ep->fds[i], &ex)) e |= EPOLLERR;
        if (e) { out[n].events = e; out[n].data.ptr = ep->ptr[i]; n++; }
    }
    return n;
}

/* accept4(fd, addr, len, flags) — flags ignored; result is set non-blocking */
static inline int accept4(int fd, struct sockaddr *addr, socklen_t *len, int flags) {
    (void)flags;
    SOCKET c = accept((SOCKET)fd, addr, len);
    if (c == INVALID_SOCKET) { errno = kci_wsa_errno(); return -1; }
    u_long m = 1; ioctlsocket(c, FIONBIO, &m);
    return (int)c;
}

#endif /* KCI_WIN_SOCKETS */
#endif /* _WIN32 */
#endif /* KCI_WIN_COMPAT_H */
