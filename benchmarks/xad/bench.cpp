// XAD (tape-based AAD with expression templates) on the comparison workloads of
// ../src/lib.rs: same generator, same inputs, f64, single core. See ../README.md to build.
#include <XAD/XAD.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

constexpr int MC_PATHS = 200000, MC_STEPS = 52, MC_W = 16, F_POINTS = 1 << 20, REPS = 31;
constexpr double MC_PARAMS[5] = {100.0, 0.01, 0.2, 1.0, 100.0};

using mode = xad::adj<double>;
using AD = mode::active_type;
using Tape = mode::tape_type;

struct Rng {  // splitmix64
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double uniform() { return (double(next() >> 11) + 0.5) * (1.0 / double(1ULL << 53)); }
};

std::vector<float> normals() {
    Rng rng{7};
    std::vector<float> z;
    z.reserve(size_t(MC_PATHS) * MC_STEPS);
    while (z.size() < size_t(MC_PATHS) * MC_STEPS) {
        double r = std::sqrt(-2.0 * std::log(rng.uniform()));
        double t = 6.283185307179586 * rng.uniform();
        z.push_back(float(r * std::cos(t)));
        z.push_back(float(r * std::sin(t)));
    }
    return z;
}

// Path-wise adjoints: record the setup once; per path record, propagate back to the mark and reset;
// one final pass through the setup.
double mc(Tape& tape, const std::vector<float>& z, double g[5]) {
    AD p[5];
    for (int i = 0; i < 5; ++i) p[i] = MC_PARAMS[i], tape.registerInput(p[i]);
    tape.newRecording();
    AD dt = p[3] / double(MC_STEPS);
    AD drift = (p[1] - 0.5 * p[2] * p[2]) * dt;
    AD vol = p[2] * sqrt(dt);
    AD x0 = log(p[0]);
    AD disc = exp(-p[1] * p[3]);
    auto mark = tape.getPosition();
    double acc = 0;
    for (int i = 0; i < MC_PATHS / MC_W; ++i)
        for (int l = 0; l < MC_W; ++l) {
            const float* zp = z.data() + size_t(i) * MC_STEPS * MC_W + l;
            AD x = x0;
            for (int k = 0; k < MC_STEPS; ++k) x = x + drift + vol * double(zp[k * MC_W]);
            AD v = exp(x) - p[4];
            // passive mask: a ternary on active values leaks a tape slot per path
            AD pay = v * disc * (xad::value(v) > 0 ? 1.0 : 0.0);
            acc += xad::value(pay);
            tape.registerOutput(pay);
            xad::derivative(pay) = 1.0 / MC_PATHS;
            tape.computeAdjointsTo(mark);
            tape.clearDerivativesAfter(mark);
            tape.resetTo(mark);
        }
    tape.computeAdjoints();
    for (int i = 0; i < 5; ++i) g[i] = xad::derivative(p[i]);
    tape.clearAll();
    return acc / MC_PATHS;
}

void grad_f(Tape& tape, const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z,
            std::vector<double>& v, std::vector<double>& gx, std::vector<double>& gy) {
    for (size_t i = 0; i < x.size(); ++i) {
        AD xa = x[i], ya = y[i];
        tape.registerInput(xa);
        tape.registerInput(ya);
        tape.newRecording();
        AD f = exp(xa * ya) + sqrt(xa * xa + z[i]) / ya;
        tape.registerOutput(f);
        xad::derivative(f) = 1.0;
        tape.computeAdjoints();
        v[i] = xad::value(f), gx[i] = xad::derivative(xa), gy[i] = xad::derivative(ya);
        tape.clearAll();
    }
}

template <class F>
double best_ms(F f) {
    double best = 1e30;
    volatile double sink = 0;
    for (int i = 0; i < REPS; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        sink = sink + f();
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

double sum(const std::vector<double>& a) {
    double s = 0;
    for (double x : a) s += x;
    return s;
}

int main() {
    Tape tape;
    auto z = normals();
    Rng rng{11};
    std::vector<double> x(F_POINTS), y(F_POINTS), zd(F_POINTS), v(F_POINTS), gx(F_POINTS), gy(F_POINTS);
    for (int i = 0; i < F_POINTS; ++i) x[i] = 0.1 + rng.uniform(), y[i] = 1.0 + rng.uniform(), zd[i] = rng.uniform();

    double g[5];
    double p = mc(tape, z, g);
    std::printf("%-8s mc  price %.10f grad [%.10f, %.10f, %.10f, %.10f, %.10f]\n", "xad", p, g[0], g[1], g[2], g[3], g[4]);
    grad_f(tape, x, y, zd, v, gx, gy);
    std::printf("%-8s f   sum f %.10f sum df/dx %.10f sum df/dy %.10f\n", "xad", sum(v), sum(gx), sum(gy));

    std::printf("%-8s %-32s %9.3f ms\n", "xad", "mc price + 5 Greeks", best_ms([&] {
        double g[5];
        double p = mc(tape, z, g);
        return p + g[0] + g[1] + g[2] + g[3] + g[4];
    }));
    std::printf("%-8s %-32s %9.3f ms\n", "xad", "f value + gradient", best_ms([&] {
        grad_f(tape, x, y, zd, v, gx, gy);
        return v.back() + gx.back() + gy.back();
    }));
}
