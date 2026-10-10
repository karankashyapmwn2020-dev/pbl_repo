#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace memtrack {
inline std::size_t cur = 0;
inline std::size_t peak = 0;
constexpr std::size_t HDR = 16;
#ifdef ARENA_NO_MEMTRACK
constexpr bool enabled = false;
#else
constexpr bool enabled = true;
#endif
}

#ifndef ARENA_NO_MEMTRACK
void* operator new(std::size_t n) {
    void* p = std::malloc(n + memtrack::HDR);
    if (!p) throw std::bad_alloc();
    *static_cast<std::size_t*>(p) = n;
    memtrack::cur += n;
    if (memtrack::cur > memtrack::peak) memtrack::peak = memtrack::cur;
    return static_cast<char*>(p) + memtrack::HDR;
}
void operator delete(void* p) noexcept {
    if (!p) return;

    void* base = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(p) - memtrack::HDR);
    memtrack::cur -= *static_cast<std::size_t*>(base);
    std::free(base);
}
void operator delete(void* p, std::size_t) noexcept { operator delete(p); }
#endif

struct Ops {
    std::uint64_t cmp = 0;
    std::uint64_t swp = 0;
    std::uint64_t vis = 0;
    std::uint64_t rel = 0;
    std::uint64_t dp = 0;
    std::uint64_t total() const { return cmp + swp + vis + rel + dp; }
    void reset() { *this = Ops(); }
};
inline Ops g_ops;

struct Rng {
    std::uint64_t s;
    explicit Rng(std::uint64_t seed) : s(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    long long range(long long lo, long long hi) {
        return lo + static_cast<long long>(next() % static_cast<std::uint64_t>(hi - lo + 1));
    }
    double unit() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
};

using Clock = std::chrono::steady_clock;
using VecI = std::vector<int>;

struct AlgoInfo {
    std::string name, complexity;
    bool exact;
};

struct RunConfig {
    int scenario = 0;
    long long n = 0;
    std::uint64_t seed = 42;
    int reps = 5;
    std::vector<int> selected;
};

struct Verdict {
    bool ok = true;
    std::string msg;
    double quality = 100.0;
};

struct Row {
    std::string name, complexity, status;
    bool exact = true, valid = false;
    double medMs = 0, minMs = 0, quality = 100, score = 0;
    Ops ops;
    std::size_t mem = 0;
    int rank = 0;
};

struct Aborted : std::runtime_error {
    using std::runtime_error::runtime_error;
};

template <class I, class O>
struct Algo {
    std::string name, complexity, limit;
    bool exact = true;
    bool mutates = false;
    std::function<bool(const I&)> eligible;
    std::function<O(I&)> run;
};

template <class I, class O>
Algo<I, O> mk(std::string name, std::string cx, std::string limit, bool exact, bool mutates,
              std::function<bool(const I&)> elig, std::function<O(I&)> run) {
    Algo<I, O> a;
    a.name = std::move(name);
    a.complexity = std::move(cx);
    a.limit = std::move(limit);
    a.exact = exact;
    a.mutates = mutates;
    a.eligible = std::move(elig);
    a.run = std::move(run);
    return a;
}

template <class I, class O>
std::vector<AlgoInfo> infoOf(const std::vector<Algo<I, O>>& v) {
    std::vector<AlgoInfo> r;
    for (const auto& a : v) r.push_back({a.name, a.complexity, a.exact});
    return r;
}

template <class I, class O>
std::vector<Row> benchmark(I& inst, const std::vector<Algo<I, O>>& algos, const std::vector<int>& selected,
                           int reps, const std::function<Verdict(const O&, bool)>& validate) {
    std::vector<Row> rows;
    for (std::size_t k = 0; k < algos.size(); ++k) {
        if (!selected.empty() && std::find(selected.begin(), selected.end(), static_cast<int>(k)) == selected.end())
            continue;
        const Algo<I, O>& a = algos[k];
        Row row;
        row.name = a.name;
        row.complexity = a.complexity;
        row.exact = a.exact;

        if (a.eligible && !a.eligible(inst)) {
            row.status = "SKIPPED (limit: " + a.limit + ")";
            rows.push_back(row);
            continue;
        }

        std::vector<double> times;
        Verdict verdict;
        try {
            for (int r = 0; r < reps; ++r) {
                std::unique_ptr<I> work;
                I* target = &inst;
                if (a.mutates) {
                    work.reset(new I(inst));
                    target = work.get();
                }
                g_ops.reset();
                const std::size_t base = memtrack::cur;
                memtrack::peak = base;
                const auto t0 = Clock::now();
                O out = a.run(*target);
                const auto t1 = Clock::now();
                const std::size_t peakDelta = memtrack::peak - base;
                times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                if (r == 0) {
                    row.ops = g_ops;
                    row.mem = peakDelta;
                    verdict = validate(out, a.exact);
                }
            }
        } catch (const Aborted& e) {
            row.status = std::string("ABORTED (") + e.what() + ")";
            rows.push_back(row);
            continue;
        } catch (const std::exception& e) {
            row.status = std::string("ERROR: ") + e.what();
            rows.push_back(row);
            continue;
        }

        std::sort(times.begin(), times.end());
        const std::size_t m = times.size();
        row.medMs = (m % 2) ? times[m / 2] : 0.5 * (times[m / 2 - 1] + times[m / 2]);
        row.minMs = times.front();
        row.valid = verdict.ok;
        row.quality = verdict.quality;
        if (!verdict.ok) row.status = "FAIL: " + verdict.msg;
        else row.status = a.exact ? "OK" : "OK (heuristic)";
        rows.push_back(row);
    }
    return rows;
}

void analyze(std::vector<Row>& rows, bool qualityFirst) {
    std::vector<Row*> v;
    for (auto& r : rows)
        if (r.valid) v.push_back(&r);
    std::sort(v.begin(), v.end(), [&](const Row* a, const Row* b) {
        if (qualityFirst && std::fabs(a->quality - b->quality) > 1e-9) return a->quality > b->quality;
        return a->medMs < b->medMs;
    });
    double best = 1e300;
    for (auto* r : v) best = std::min(best, std::max(r->medMs, 1e-9));
    int rk = 1;
    for (auto* r : v) {
        r->rank = rk++;
        r->score = r->quality * best / std::max(r->medMs, 1e-9);
    }
    std::stable_sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        if (a.valid != b.valid) return a.valid;
        if (a.valid) return a.rank < b.rank;
        return false;
    });
}

namespace sorting {
inline bool lt(int a, int b) { ++g_ops.cmp; return a < b; }
inline void sw(int& a, int& b) { std::swap(a, b); ++g_ops.swp; }

void bubble(VecI& a) {
    const std::size_t n = a.size();
    for (std::size_t i = 0; i + 1 < n; ++i) {
        bool swapped = false;
        for (std::size_t j = 0; j + 1 < n - i; ++j)
            if (lt(a[j + 1], a[j])) { sw(a[j], a[j + 1]); swapped = true; }
        if (!swapped) break;
    }
}
void selection(VecI& a) {
    const std::size_t n = a.size();
    for (std::size_t i = 0; i + 1 < n; ++i) {
        std::size_t m = i;
        for (std::size_t j = i + 1; j < n; ++j)
            if (lt(a[j], a[m])) m = j;
        if (m != i) sw(a[i], a[m]);
    }
}
void insertion(VecI& a) {
    for (std::size_t i = 1; i < a.size(); ++i) {
        int key = a[i];
        std::size_t j = i;
        while (j > 0 && lt(key, a[j - 1])) { a[j] = a[j - 1]; ++g_ops.swp; --j; }
        a[j] = key;
    }
}
void shell(VecI& a) {
    const std::size_t n = a.size();
    std::size_t gap = 1;
    while (gap < n / 3) gap = gap * 3 + 1;
    for (; gap > 0; gap /= 3)
        for (std::size_t i = gap; i < n; ++i) {
            int key = a[i];
            std::size_t j = i;
            while (j >= gap && lt(key, a[j - gap])) { a[j] = a[j - gap]; ++g_ops.swp; j -= gap; }
            a[j] = key;
        }
}
void mergeRec(VecI& a, VecI& t, std::size_t lo, std::size_t hi) {
    if (hi - lo < 2) return;
    const std::size_t mid = lo + (hi - lo) / 2;
    mergeRec(a, t, lo, mid);
    mergeRec(a, t, mid, hi);
    std::size_t i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        if (lt(a[j], a[i])) t[k++] = a[j++]; else t[k++] = a[i++];
        ++g_ops.swp;
    }
    while (i < mid) { t[k++] = a[i++]; ++g_ops.swp; }
    while (j < hi) { t[k++] = a[j++]; ++g_ops.swp; }
    for (k = lo; k < hi; ++k) a[k] = t[k];
}
void mergeSort(VecI& a) {
    VecI tmp(a.size());
    mergeRec(a, tmp, 0, a.size());
}
void quickRec(VecI& a, long long lo, long long hi) {
    while (lo < hi) {
        const long long mid = lo + (hi - lo) / 2;
        if (lt(a[mid], a[lo])) sw(a[mid], a[lo]);
        if (lt(a[hi], a[lo])) sw(a[hi], a[lo]);
        if (lt(a[hi], a[mid])) sw(a[hi], a[mid]);
        const int pivot = a[mid];
        long long i = lo, j = hi;
        while (i <= j) {
            while (lt(a[i], pivot)) ++i;
            while (lt(pivot, a[j])) --j;
            if (i <= j) {
                if (i != j) sw(a[i], a[j]);
                ++i;
                --j;
            }
        }
        if (j - lo < hi - i) { quickRec(a, lo, j); lo = i; }
        else { quickRec(a, i, hi); hi = j; }
    }
}
void quick(VecI& a) { if (a.size() > 1) quickRec(a, 0, static_cast<long long>(a.size()) - 1); }
void siftDown(VecI& a, std::size_t root, std::size_t n) {
    for (;;) {
        std::size_t c = 2 * root + 1;
        if (c >= n) return;
        if (c + 1 < n && lt(a[c], a[c + 1])) ++c;
        if (lt(a[root], a[c])) { sw(a[root], a[c]); root = c; } else return;
    }
}
void heapSort(VecI& a) {
    const std::size_t n = a.size();
    if (n < 2) return;
    for (std::size_t i = n / 2; i-- > 0;) siftDown(a, i, n);
    for (std::size_t e = n - 1; e > 0; --e) { sw(a[0], a[e]); siftDown(a, 0, e); }
}
void stdSort(VecI& a) {
    std::sort(a.begin(), a.end(), [](int x, int y) { ++g_ops.cmp; return x < y; });
}

using SortFn = void (*)(VecI&);
std::function<VecI(VecI&)> runner(SortFn f) {
    return [f](VecI& a) -> VecI { f(a); return std::move(a); };
}
}

std::vector<Algo<VecI, VecI>> sortAlgos() {
    using namespace sorting;
    auto small = [](const VecI& v) { return v.size() <= 30000; };
    std::vector<Algo<VecI, VecI>> r;
    r.push_back(mk<VecI, VecI>("Bubble Sort", "O(n^2)", "n<=30000", true, true, small, runner(bubble)));
    r.push_back(mk<VecI, VecI>("Selection Sort", "O(n^2)", "n<=30000", true, true, small, runner(selection)));
    r.push_back(mk<VecI, VecI>("Insertion Sort", "O(n^2)", "n<=30000", true, true, small, runner(insertion)));
    r.push_back(mk<VecI, VecI>("Shell Sort (Knuth)", "~O(n^1.5)", "", true, true, nullptr, runner(shell)));
    r.push_back(mk<VecI, VecI>("Merge Sort", "O(n log n)", "", true, true, nullptr, runner(mergeSort)));
    r.push_back(mk<VecI, VecI>("Quick Sort (median-of-3)", "O(n log n) avg", "", true, true, nullptr, runner(quick)));
    r.push_back(mk<VecI, VecI>("Heap Sort", "O(n log n)", "", true, true, nullptr, runner(heapSort)));
    r.push_back(mk<VecI, VecI>("std::sort (introsort)", "O(n log n)", "", true, true, nullptr, runner(stdSort)));
    return r;
}

VecI genSort(int sc, std::size_t n, std::uint64_t seed) {
    Rng r(seed);
    VecI a(n);
    auto fill = [&](long long lo, long long hi) { for (auto& x : a) x = static_cast<int>(r.range(lo, hi)); };
    switch (sc) {
        case 0: fill(0, 1000000000); break;
        case 1: fill(0, 1000000000); std::sort(a.begin(), a.end()); break;
        case 2: fill(0, 1000000000); std::sort(a.begin(), a.end(), std::greater<int>()); break;
        case 3: {
            fill(0, 1000000000);
            std::sort(a.begin(), a.end());
            for (std::size_t k = 0; k < n / 100 + 1 && n > 1; ++k)
                std::swap(a[r.range(0, n - 1)], a[r.range(0, n - 1)]);
            break;
        }
        case 4: fill(0, 9); break;
        default:
            for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<int>(i < n / 2 ? i : n - 1 - i);
    }
    return a;
}

struct SearchInst {
    VecI a;
    VecI q;
};

namespace searching {
int linear(const VecI& a, int t) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        ++g_ops.cmp;
        if (a[i] == t) return static_cast<int>(i);
    }
    return -1;
}
int binRange(const VecI& a, long long lo, long long hi, int t) {
    while (lo <= hi) {
        const long long mid = lo + (hi - lo) / 2;
        ++g_ops.cmp;
        if (a[mid] == t) return static_cast<int>(mid);
        ++g_ops.cmp;
        if (a[mid] < t) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}
int binary(const VecI& a, int t) { return binRange(a, 0, static_cast<long long>(a.size()) - 1, t); }
int jump(const VecI& a, int t) {
    const std::size_t n = a.size();
    const std::size_t step = std::max<std::size_t>(1, static_cast<std::size_t>(std::sqrt(static_cast<double>(n))));
    std::size_t prev = 0;
    while (prev < n && (++g_ops.cmp, a[std::min(prev + step, n) - 1] < t)) prev += step;
    if (prev >= n) return -1;
    const std::size_t end = std::min(prev + step, n);
    for (std::size_t i = prev; i < end; ++i) {
        ++g_ops.cmp;
        if (a[i] == t) return static_cast<int>(i);
        if (a[i] > t) return -1;
    }
    return -1;
}
int interpolation(const VecI& a, int t) {
    long long lo = 0, hi = static_cast<long long>(a.size()) - 1;
    while (lo <= hi && t >= a[lo] && t <= a[hi]) {
        ++g_ops.cmp;
        if (a[hi] == a[lo]) return a[lo] == t ? static_cast<int>(lo) : -1;
        long long pos = lo + static_cast<long long>(static_cast<double>(hi - lo) *
                                                    (static_cast<double>(t) - a[lo]) /
                                                    (static_cast<double>(a[hi]) - a[lo]));
        pos = std::max(lo, std::min(hi, pos));
        ++g_ops.cmp;
        if (a[pos] == t) return static_cast<int>(pos);
        if (a[pos] < t) lo = pos + 1; else hi = pos - 1;
    }
    return -1;
}
int exponential(const VecI& a, int t) {
    const long long n = static_cast<long long>(a.size());
    ++g_ops.cmp;
    if (a[0] == t) return 0;
    long long i = 1;
    while (i < n && (++g_ops.cmp, a[i] < t)) i *= 2;
    return binRange(a, i / 2, std::min(i, n - 1), t);
}
int stdLower(const VecI& a, int t) {
    auto it = std::lower_bound(a.begin(), a.end(), t, [](int x, int y) { ++g_ops.cmp; return x < y; });
    ++g_ops.cmp;
    return (it != a.end() && *it == t) ? static_cast<int>(it - a.begin()) : -1;
}
using SearchFn = int (*)(const VecI&, int);
std::function<VecI(SearchInst&)> runner(SearchFn f) {
    return [f](SearchInst& s) {
        VecI out(s.q.size());
        for (std::size_t i = 0; i < s.q.size(); ++i) out[i] = f(s.a, s.q[i]);
        return out;
    };
}
}

std::vector<Algo<SearchInst, VecI>> searchAlgos() {
    using namespace searching;
    using A = SearchInst;
    auto okLinear = [](const A& s) { return static_cast<double>(s.a.size()) * s.q.size() <= 2e8; };
    std::vector<Algo<A, VecI>> r;
    r.push_back(mk<A, VecI>("Linear Search", "O(n)", "n*queries<=2e8", true, false, okLinear, runner(linear)));
    r.push_back(mk<A, VecI>("Binary Search", "O(log n)", "", true, false, nullptr, runner(binary)));
    r.push_back(mk<A, VecI>("Jump Search", "O(sqrt n)", "", true, false, nullptr, runner(jump)));
    r.push_back(mk<A, VecI>("Interpolation Search", "O(log log n)*", "", true, false, nullptr, runner(interpolation)));
    r.push_back(mk<A, VecI>("Exponential Search", "O(log n)", "", true, false, nullptr, runner(exponential)));
    r.push_back(mk<A, VecI>("std::lower_bound", "O(log n)", "", true, false, nullptr, runner(stdLower)));
    return r;
}

SearchInst genSearch(int sc, std::size_t n, std::uint64_t seed) {
    Rng r(seed);
    SearchInst s;
    s.a.resize(n);
    for (auto& x : s.a) {
        if (sc == 0) x = static_cast<int>(r.range(0, 4 * static_cast<long long>(n)));
        else if (sc == 1) x = static_cast<int>(r.range(0, std::max<long long>(1, n / 50)));
        else x = static_cast<int>(1e9 * std::pow(r.unit(), 4.0));
    }
    std::sort(s.a.begin(), s.a.end());
    const std::size_t Q = 2000;
    s.q.resize(Q);
    const long long hiVal = static_cast<long long>(s.a.back()) + 1;
    for (std::size_t i = 0; i < Q; ++i)
        s.q[i] = (i % 2 == 0) ? s.a[r.range(0, n - 1)] : static_cast<int>(r.range(0, hiVal));
    return s;
}

struct Problem {
    std::string key, title, sizeLabel;
    std::vector<std::string> scenarios;
    long long defaultN = 0, minN = 1, maxN = 0;
    bool qualityFirst = false;
    std::function<std::vector<AlgoInfo>()> info;
    std::function<std::vector<Row>(RunConfig&, std::string&)> run;
};

std::string fmtD(double v, int prec) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << v;
    return o.str();
}

std::vector<Problem> buildProblems() {
    std::vector<Problem> ps;

    {
        Problem p;
        p.key = "sort"; p.title = "Sorting"; p.sizeLabel = "number of elements";
        p.scenarios = {"random", "sorted", "reverse-sorted", "nearly-sorted", "duplicate-heavy", "organ-pipe (stress)"};
        p.defaultN = 10000; p.minN = 1; p.maxN = 10000000;
        p.info = [] { return infoOf(sortAlgos()); };
        p.run = [](RunConfig& c, std::string& desc) {
            VecI inst = genSort(c.scenario, static_cast<std::size_t>(c.n), c.seed);
            VecI ref = inst;
            std::sort(ref.begin(), ref.end());
            desc = "n=" + std::to_string(inst.size()) + " integers";
            auto algos = sortAlgos();
            return benchmark<VecI, VecI>(inst, algos, c.selected, c.reps, [&](const VecI& out, bool) -> Verdict {
                if (out == ref) return Verdict{};
                return Verdict{false, "output is not the sorted permutation of the input", 0.0};
            });
        };
        ps.push_back(p);
    }
    {
        Problem p;
        p.key = "search"; p.title = "Searching (2000 queries on a sorted array)"; p.sizeLabel = "array size";
        p.scenarios = {"uniform", "duplicate-heavy", "skewed (hard for interpolation)"};
        p.defaultN = 1000000; p.minN = 1; p.maxN = 10000000;
        p.info = [] { return infoOf(searchAlgos()); };
        p.run = [](RunConfig& c, std::string& desc) {
            SearchInst inst = genSearch(c.scenario, static_cast<std::size_t>(c.n), c.seed);
            std::vector<char> present(inst.q.size());
            for (std::size_t i = 0; i < inst.q.size(); ++i)
                present[i] = std::binary_search(inst.a.begin(), inst.a.end(), inst.q[i]);
            desc = "array size=" + std::to_string(inst.a.size()) + ", queries=" + std::to_string(inst.q.size());
            auto algos = searchAlgos();
            return benchmark<SearchInst, VecI>(inst, algos, c.selected, c.reps, [&](const VecI& out, bool) -> Verdict {
                if (out.size() != inst.q.size()) return Verdict{false, "wrong number of answers", 0.0};
                for (std::size_t i = 0; i < out.size(); ++i) {
                    const int r = out[i];
                    if (r < 0) {
                        if (present[i]) return Verdict{false, "missed key " + std::to_string(inst.q[i]), 0.0};
                    } else if (static_cast<std::size_t>(r) >= inst.a.size() || inst.a[r] != inst.q[i]) {
                        return Verdict{false, "wrong index for key " + std::to_string(inst.q[i]), 0.0};
                    }
                }
                return Verdict{};
            });
        };
        ps.push_back(p);
    }
    return ps;
}

std::string fmtCount(std::uint64_t v) {
    char b[32];
    if (v < 10000) std::snprintf(b, sizeof b, "%llu", static_cast<unsigned long long>(v));
    else if (v < 1000000) std::snprintf(b, sizeof b, "%.1fK", v / 1e3);
    else if (v < 1000000000) std::snprintf(b, sizeof b, "%.2fM", v / 1e6);
    else std::snprintf(b, sizeof b, "%.2fG", v / 1e9);
    return b;
}
std::string fmtBytes(std::size_t v) {
    char b[32];
    if (v < 1024) std::snprintf(b, sizeof b, "%zu B", v);
    else if (v < 1024 * 1024) std::snprintf(b, sizeof b, "%.1f KB", v / 1024.0);
    else if (v < 1024ULL * 1024 * 1024) std::snprintf(b, sizeof b, "%.1f MB", v / 1048576.0);
    else std::snprintf(b, sizeof b, "%.2f GB", v / 1073741824.0);
    return b;
}
std::string memStr(std::size_t v) { return memtrack::enabled ? fmtBytes(v) : std::string("n/a"); }
std::string fit(const std::string& s, std::size_t w) { return s.size() <= w ? s : s.substr(0, w - 1) + "~"; }

void printDashboard(const Problem& p, const RunConfig& c, const std::string& desc, const std::vector<Row>& rows) {
    const int W = 118;
    std::cout << "\n" << std::string(W, '=') << "\n";
    std::cout << " ALGORITHM BATTLE ARENA | " << p.title << " | scenario: " << p.scenarios[c.scenario] << "\n";
    std::cout << " " << desc << " | seed=" << c.seed << " | repetitions=" << c.reps << " (median reported)\n";
    std::cout << std::string(W, '=') << "\n";
    std::cout << std::left << std::setw(5) << "Rank" << std::setw(28) << "Algorithm" << std::setw(22) << "Complexity"
              << std::right << std::setw(12) << "Median(ms)" << std::setw(11) << "Min(ms)" << std::setw(10) << "Ops"
              << std::setw(10) << "Aux mem" << std::setw(9) << "Quality" << std::setw(8) << "Score" << "  Status\n";
    std::cout << std::string(W, '-') << "\n";
    for (const Row& r : rows) {
        std::cout << std::left << std::setw(5) << (r.valid ? std::to_string(r.rank) : std::string("-"))
                  << std::setw(28) << fit(r.name, 27) << std::setw(22) << fit(r.complexity, 21) << std::right;
        if (r.valid || r.medMs > 0) {
            std::cout << std::setw(12) << fmtD(r.medMs, 3) << std::setw(11) << fmtD(r.minMs, 3) << std::setw(10)
                      << fmtCount(r.ops.total()) << std::setw(10) << memStr(r.mem) << std::setw(9)
                      << (fmtD(r.quality, 1) + "%") << std::setw(8) << (r.valid ? fmtD(r.score, 1) : "-");
        } else {
            std::cout << std::setw(12) << "--" << std::setw(11) << "--" << std::setw(10) << "--" << std::setw(10)
                      << "--" << std::setw(9) << "--" << std::setw(8) << "--";
        }
        std::cout << "  " << r.status << "\n";
    }
    std::cout << std::string(W, '-') << "\n";

    std::vector<const Row*> v;
    for (const Row& r : rows)
        if (r.valid) v.push_back(&r);
    if (v.empty()) { std::cout << " No algorithm produced a validated result.\n"; return; }

    if (v.size() >= 2) {
        double tmin = 1e300, tmax = 0;
        for (auto* r : v) { tmin = std::min(tmin, std::max(r->medMs, 1e-9)); tmax = std::max(tmax, std::max(r->medMs, 1e-9)); }
        std::cout << " Performance graph - median time (log scale, longer bar = slower)\n";
        for (auto* r : v) {
            int len = 40;
            if (tmax > tmin * 1.0001)
                len = 1 + static_cast<int>(std::lround(39.0 * std::log(std::max(r->medMs, 1e-9) / tmin) / std::log(tmax / tmin)));
            std::cout << "  " << std::left << std::setw(27) << fit(r->name, 26) << std::string(len, '#') << " "
                      << fmtD(r->medMs, 3) << " ms\n";
        }
    }

    const Row* fastest = v[0];
    const Row* fewest = v[0];
    const Row* leanest = v[0];
    for (auto* r : v) {
        if (r->medMs < fastest->medMs) fastest = r;
        if (r->ops.total() < fewest->ops.total()) fewest = r;
        if (r->mem < leanest->mem) leanest = r;
    }
    std::cout << "\n Recommendations\n";
    std::cout << "  * Overall winner (rank 1): " << v[0]->name << "\n";
    std::cout << "  * Fastest validated run  : " << fastest->name << " (" << fmtD(fastest->medMs, 3) << " ms)\n";
    std::cout << "  * Fewest operations      : " << fewest->name << " (" << fmtCount(fewest->ops.total()) << ")\n";
    if (memtrack::enabled)
        std::cout << "  * Lowest auxiliary memory: " << leanest->name << " (" << fmtBytes(leanest->mem) << ")\n";
    for (auto* r : v)
        if (!r->exact && r->quality < 99.999)
            std::cout << "  ! " << r->name << " is a heuristic and reached only " << fmtD(r->quality, 1) << "% of the optimum.\n";
    for (const Row& r : rows)
        if (r.status.rfind("FAIL", 0) == 0) std::cout << "  ! " << r.name << " failed validation and was excluded from ranking.\n";
}

std::string csvQ(const std::string& s) {
    std::string o = "\"";
    for (char ch : s) { if (ch == '"') o += '"'; o += ch; }
    return o + "\"";
}

void exportCsv(const std::string& path, const Problem& p, const RunConfig& c, const std::vector<Row>& rows) {
    const bool fresh = !std::ifstream(path).good();
    std::ofstream f(path, std::ios::app);
    if (!f) { std::cerr << "warning: cannot write " << path << "\n"; return; }
    if (fresh)
        f << "problem,scenario,n,seed,repetitions,algorithm,status,rank,median_ms,min_ms,comparisons,swaps_moves,"
             "visits,relaxations,dp_transitions,peak_aux_bytes,quality_pct,score\n";
    for (const Row& r : rows) {
        f << p.key << "," << csvQ(p.scenarios[c.scenario]) << "," << c.n << "," << c.seed << "," << c.reps << ","
          << csvQ(r.name) << "," << csvQ(r.status) << ",";
        if (r.valid || r.medMs > 0) {
            f << (r.valid ? std::to_string(r.rank) : "") << "," << fmtD(r.medMs, 6) << "," << fmtD(r.minMs, 6) << ","
              << r.ops.cmp << "," << r.ops.swp << "," << r.ops.vis << "," << r.ops.rel << "," << r.ops.dp << ","
              << (memtrack::enabled ? std::to_string(r.mem) : std::string()) << "," << fmtD(r.quality, 3) << "," << (r.valid ? fmtD(r.score, 3) : "") << "\n";
        } else {
            f << ",,,,,,,,,,\n";
        }
    }
}

std::string lower(std::string s) {
    for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}
std::string trim(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

RunConfig clampConfig(const Problem& p, RunConfig c) {
    c.n = std::max(p.minN, std::min(p.maxN, c.n));
    c.reps = std::max(1, std::min(1000, c.reps));
    return c;
}

std::vector<Row> execute(const Problem& p, RunConfig c, const std::string& csv, bool show = true) {
    c = clampConfig(p, c);
    std::cout << "\nRunning '" << p.title << "' / " << p.scenarios[c.scenario] << " ..." << std::endl;
    std::string desc;
    std::vector<Row> rows = p.run(c, desc);
    analyze(rows, p.qualityFirst);
    if (show) printDashboard(p, c, desc, rows);
    if (!csv.empty()) exportCsv(csv, p, c, rows);
    return rows;
}

void runSuite(const Problem& p, int reps, std::uint64_t seed, const std::string& csv) {
    std::vector<std::vector<Row>> all;
    for (std::size_t s = 0; s < p.scenarios.size(); ++s) {
        RunConfig c;
        c.scenario = static_cast<int>(s);
        c.n = p.defaultN;
        c.seed = seed;
        c.reps = reps;
        all.push_back(execute(p, c, csv));
    }

    std::cout << "\n" << std::string(118, '#') << "\n SUMMARY - " << p.title << ": median time (ms) per scenario   [* = fastest validated in column]\n"
              << std::string(118, '#') << "\n";
    std::cout << std::left << std::setw(28) << "Algorithm";
    for (const auto& s : p.scenarios) std::cout << std::right << std::setw(16) << fit(s, 15);
    std::cout << "\n";
    std::vector<double> colBest(p.scenarios.size(), 1e300);
    for (std::size_t s = 0; s < all.size(); ++s)
        for (const Row& r : all[s])
            if (r.valid && r.rank == 1) colBest[s] = r.medMs;
    for (std::size_t s = 0; s < all.size(); ++s)
        for (const Row& r : all[s])
            if (r.valid) colBest[s] = std::min(colBest[s], r.medMs);
    for (const AlgoInfo& ai : p.info()) {
        std::cout << std::left << std::setw(28) << fit(ai.name, 27);
        for (std::size_t s = 0; s < all.size(); ++s) {
            std::string cell = "n/a";
            for (const Row& r : all[s])
                if (r.name == ai.name) {
                    if (r.valid) cell = fmtD(r.medMs, 3) + (r.medMs <= colBest[s] * 1.0000001 ? "*" : "");
                    else if (r.status.rfind("SKIPPED", 0) == 0) cell = "skipped";
                    else if (r.status.rfind("ABORTED", 0) == 0) cell = "aborted";
                    else cell = "FAIL";
                }
            std::cout << std::right << std::setw(16) << cell;
        }
        std::cout << "\n";
    }
}

bool parseSelection(const std::vector<AlgoInfo>& info, const std::string& text, std::vector<int>& out, std::string& err) {
    out.clear();
    const std::string t = lower(trim(text));
    if (t.empty() || t == "all") return true;
    std::stringstream ss(t);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok = trim(tok);
        if (tok.empty()) continue;
        int idx = -1;
        if (std::all_of(tok.begin(), tok.end(), [](unsigned char ch) { return std::isdigit(ch); })) {
            const int k = std::atoi(tok.c_str());
            if (k >= 1 && k <= static_cast<int>(info.size())) idx = k - 1;
        } else {
            for (std::size_t i = 0; i < info.size(); ++i)
                if (lower(info[i].name).find(tok) != std::string::npos) { idx = static_cast<int>(i); break; }
        }
        if (idx < 0) { err = "unknown algorithm '" + tok + "'"; return false; }
        if (std::find(out.begin(), out.end(), idx) == out.end()) out.push_back(idx);
    }
    return true;
}

int findProblem(const std::vector<Problem>& ps, const std::string& key) {
    const std::string k = lower(trim(key));
    for (std::size_t i = 0; i < ps.size(); ++i)
        if (ps[i].key == k || std::to_string(i + 1) == k) return static_cast<int>(i);
    return -1;
}
int findScenario(const Problem& p, const std::string& text) {
    const std::string t = lower(trim(text));
    if (t.empty()) return -1;
    if (std::all_of(t.begin(), t.end(), [](unsigned char ch) { return std::isdigit(ch); })) {
        const int k = std::atoi(t.c_str());
        return (k >= 1 && k <= static_cast<int>(p.scenarios.size())) ? k - 1 : -1;
    }
    for (std::size_t i = 0; i < p.scenarios.size(); ++i)
        if (lower(p.scenarios[i]).find(t) != std::string::npos) return static_cast<int>(i);
    return -1;
}

void listAll(const std::vector<Problem>& ps) {
    for (std::size_t i = 0; i < ps.size(); ++i) {
        const Problem& p = ps[i];
        std::cout << "\n" << (i + 1) << ") [" << p.key << "] " << p.title << "   (size = " << p.sizeLabel << ", default "
                  << p.defaultN << ", max " << p.maxN << ")\n   scenarios: ";
        for (std::size_t s = 0; s < p.scenarios.size(); ++s) std::cout << (s + 1) << "=" << p.scenarios[s] << (s + 1 < p.scenarios.size() ? "; " : "");
        std::cout << "\n   algorithms:\n";
        const auto info = p.info();
        for (std::size_t a = 0; a < info.size(); ++a)
            std::cout << "     " << (a + 1) << ". " << std::left << std::setw(28) << info[a].name << std::setw(24) << info[a].complexity
                      << (info[a].exact ? "exact" : "heuristic") << "\n";
    }
    std::cout << "\n";
}

void printHelp() {
    std::cout <<
        "Algorithm Battle Arena - compare algorithms fairly on identical inputs.\n\n"
        "Usage:\n"
        "  arena                         interactive menu\n"
        "  arena --list                  list problems, scenarios and algorithms\n"
        "  arena --problem P [options]   run one battle\n"
        "  arena --suite [P|all]         run every scenario of one problem (or of all problems)\n\n"
        "Options:\n"
        "  --problem P    sort | search   (or 1..2)\n"
        "  --scenario S   scenario number or name fragment (default 1)\n"
        "  --n N          input size (default depends on the problem)\n"
        "  --seed K       random seed (default 42)  - same seed => same input\n"
        "  --reps R       repetitions per algorithm (default 5)\n"
        "  --algos LIST   'all' or e.g. 1,3,5 or name fragments: quick,merge\n"
        "  --csv FILE     append results to a CSV file (default arena_results.csv)\n"
        "  --nocsv        do not write a CSV file\n";
}

std::string readLine(const std::string& prompt, const std::string& def = "") {
    std::cout << prompt;
    if (!def.empty()) std::cout << " [" << def << "]";
    std::cout << ": " << std::flush;
    std::string s;
    if (!std::getline(std::cin, s)) { std::cout << "\n"; std::exit(0); }
    s = trim(s);
    return s.empty() ? def : s;
}
long long askNum(const std::string& prompt, long long def, long long lo, long long hi) {
    for (;;) {
        const std::string s = readLine(prompt + " (" + std::to_string(lo) + ".." + std::to_string(hi) + ")", std::to_string(def));
        try {
            std::size_t pos = 0;
            const long long v = std::stoll(s, &pos);
            if (pos == s.size() && v >= lo && v <= hi) return v;
        } catch (...) {}
        std::cout << "  please enter a whole number between " << lo << " and " << hi << "\n";
    }
}

int pickProblem(const std::vector<Problem>& ps) {
    std::cout << "\nProblems:\n";
    for (std::size_t i = 0; i < ps.size(); ++i) std::cout << "  " << (i + 1) << ") " << ps[i].title << "\n";
    return static_cast<int>(askNum("Choose a problem", 1, 1, static_cast<long long>(ps.size()))) - 1;
}

int interactive(const std::vector<Problem>& ps) {
    std::cout << "\n+--------------------------------------------------------------+\n"
                 "|                 A L G O R I T H M   B A T T L E   A R E N A  |\n"
                 "|   fair, reproducible, validated algorithm comparison (C++17)  |\n"
                 "+--------------------------------------------------------------+\n";
    std::string csv = readLine("CSV file for results (type 'none' to disable)", "arena_results.csv");
    if (lower(csv) == "none") csv.clear();
    for (;;) {
        std::cout << "\nMain menu\n"
                     "  1) Run a battle (choose problem, scenario, size, algorithms)\n"
                     "  2) Run all scenarios of one problem + summary matrix\n"
                     "  3) Run the full demo suite (all problems x all scenarios)\n"
                     "  4) Show problems, scenarios and algorithms\n"
                     "  0) Exit\n";
        const int choice = static_cast<int>(askNum("Your choice", 1, 0, 4));
        if (choice == 0) return 0;
        if (choice == 4) { listAll(ps); continue; }
        if (choice == 3) {
            const int reps = static_cast<int>(askNum("Repetitions", 3, 1, 100));
            const std::uint64_t seed = static_cast<std::uint64_t>(askNum("Seed", 42, 0, 2000000000LL));
            for (const Problem& p : ps) runSuite(p, reps, seed, csv);
            continue;
        }
        const Problem& p = ps[pickProblem(ps)];
        if (choice == 2) {
            const int reps = static_cast<int>(askNum("Repetitions", 5, 1, 1000));
            const std::uint64_t seed = static_cast<std::uint64_t>(askNum("Seed", 42, 0, 2000000000LL));
            runSuite(p, reps, seed, csv);
            continue;
        }
        RunConfig c;
        std::cout << "\nScenarios:\n";
        for (std::size_t s = 0; s < p.scenarios.size(); ++s) std::cout << "  " << (s + 1) << ") " << p.scenarios[s] << "\n";
        c.scenario = static_cast<int>(askNum("Choose a scenario", 1, 1, static_cast<long long>(p.scenarios.size()))) - 1;
        c.n = askNum("Input size (" + p.sizeLabel + ")", p.defaultN, p.minN, p.maxN);
        c.seed = static_cast<std::uint64_t>(askNum("Random seed", 42, 0, 2000000000LL));
        c.reps = static_cast<int>(askNum("Repetitions", 5, 1, 1000));
        const auto info = p.info();
        std::cout << "\nAlgorithms:\n";
        for (std::size_t a = 0; a < info.size(); ++a)
            std::cout << "  " << (a + 1) << ") " << std::left << std::setw(28) << info[a].name << info[a].complexity << "\n";
        for (;;) {
            std::string err;
            if (parseSelection(info, readLine("Algorithms to include (e.g. 1,3 or 'all')", "all"), c.selected, err)) break;
            std::cout << "  " << err << "\n";
        }
        execute(p, c, csv);
        if (!csv.empty()) std::cout << "\n(results appended to " << csv << ")\n";
    }
}

int runCli(const std::vector<Problem>& ps, int argc, char** argv) {
    std::string problem, scenario = "1", algos = "all", csv = "arena_results.csv", suite;
    long long n = -1;
    std::uint64_t seed = 42;
    int reps = 5;
    bool doSuite = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << name << "\n"; std::exit(1); }
            return argv[++i];
        };
        try {
            if (a == "--help" || a == "-h") { printHelp(); return 0; }
            else if (a == "--list") { listAll(ps); return 0; }
            else if (a == "--problem") problem = val("--problem");
            else if (a == "--scenario") scenario = val("--scenario");
            else if (a == "--n") n = std::stoll(val("--n"));
            else if (a == "--seed") seed = static_cast<std::uint64_t>(std::stoull(val("--seed")));
            else if (a == "--reps") reps = std::stoi(val("--reps"));
            else if (a == "--algos") algos = val("--algos");
            else if (a == "--csv") csv = val("--csv");
            else if (a == "--nocsv") csv.clear();
            else if (a == "--suite") {
                doSuite = true;
                suite = (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) ? argv[++i] : "all";
            } else { std::cerr << "unknown option: " << a << "  (try --help)\n"; return 1; }
        } catch (const std::exception&) {
            std::cerr << "invalid value for " << a << "\n";
            return 1;
        }
    }
    if (doSuite) {
        if (lower(suite) == "all") {
            for (const Problem& p : ps) runSuite(p, reps, seed, csv);
        } else {
            const int pi = findProblem(ps, suite);
            if (pi < 0) { std::cerr << "unknown problem '" << suite << "'\n"; return 1; }
            runSuite(ps[pi], reps, seed, csv);
        }
        return 0;
    }
    if (problem.empty()) { printHelp(); return 1; }
    const int pi = findProblem(ps, problem);
    if (pi < 0) { std::cerr << "unknown problem '" << problem << "' (use --list)\n"; return 1; }
    const Problem& p = ps[pi];
    RunConfig c;
    c.scenario = findScenario(p, scenario);
    if (c.scenario < 0) { std::cerr << "unknown scenario '" << scenario << "' (use --list)\n"; return 1; }
    c.n = n > 0 ? n : p.defaultN;
    c.seed = seed;
    c.reps = reps;
    std::string err;
    if (!parseSelection(p.info(), algos, c.selected, err)) { std::cerr << err << "\n"; return 1; }
    execute(p, c, csv);
    return 0;
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(true);
    const std::vector<Problem> problems = buildProblems();
    return argc > 1 ? runCli(problems, argc, argv) : interactive(problems);
}
