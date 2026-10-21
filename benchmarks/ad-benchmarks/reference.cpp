// Inputs and XAD reference results for the ad-benchmarks Heston MC and SABR calibration
// workloads, from that repository's own headers (src/heston.hpp, src/sabr.hpp), so raads runs
// on identical inputs and can be checked against XAD.
//
// Writes target/adbench.bin (little-endian f64):
//   Heston: per path, z1[100] then z2[100] (10000 paths)
//   SABR:   per expiry F, T, beta, strikes[20], market_vols[20]; initial params[15];
//           perturbation schedule[500][15]
// and prints the XAD values: Heston price and gradient; SABR objective and gradient at the
// first iteration, and the sums over all 500 iterations.

#include "src/heston.hpp"
#include "src/sabr.hpp"
#include <XAD/XAD.hpp>
#include <cstdio>

using mode = xad::adj<double>;
using AD = mode::active_type;

static void put(FILE* f, double x) { fwrite(&x, sizeof x, 1, f); }

int main()
{
    const int paths = 10000;
    FILE* f = fopen("target/adbench.bin", "wb");

    HestonInputs in;
    auto s = heston_generate_samples(paths, 77777);
    for (int p = 0; p < paths; ++p)
    {
        for (double z : s.z1[p]) put(f, z);
        for (double z : s.z2[p]) put(f, z);
    }

    SABRCalibData data;
    auto schedule = sabr_perturbation_schedule(SABR_CALIB_ITERS);
    for (auto& sl : data.slices)
    {
        put(f, sl.F);
        put(f, sl.expiry);
        put(f, sl.beta);
        for (double k : sl.strikes) put(f, k);
        for (double v : sl.market_vols) put(f, v);
    }
    double params[SABR_NUM_PARAMS];
    for (int e = 0; e < SABR_NUM_EXPIRIES; ++e)
    {
        params[3 * e] = data.init_alpha[e];
        params[3 * e + 1] = data.init_rho[e];
        params[3 * e + 2] = data.init_nu[e];
    }
    for (double p : params) put(f, p);
    for (auto& d : schedule)
        for (double x : d.dp) put(f, x);
    fclose(f);

    // Heston: pathwise adjoints, as in xad/heston_xad.cpp
    mode::tape_type tape;
    double price = 0, g[8] = {};
    for (int p = 0; p < paths; ++p)
    {
        tape.clearAll();
        AD x[8] = {in.S0, in.K, in.T, in.r, in.v0, in.kappa, in.theta, in.xi};
        for (auto& v : x) tape.registerInput(v);
        tape.newRecording();
        AD y = heston_path(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], in.rho_corr, s.z1[p], s.z2[p]);
        tape.registerOutput(y);
        derivative(y) = 1.0;
        tape.computeAdjoints();
        price += value(y);
        for (int i = 0; i < 8; ++i) g[i] += derivative(x[i]);
    }
    printf("heston %.17g", price / paths);
    for (double v : g) printf(" %.17g", v / paths);
    printf("\n");

    // SABR: objective and gradient per iteration, as in xad/sabr_calibration_xad.cpp
    double obj0 = 0, g0[SABR_NUM_PARAMS], sum_obj = 0, sum_g[SABR_NUM_PARAMS] = {};
    for (int it = 0; it < SABR_CALIB_ITERS; ++it)
    {
        tape.clearAll();
        AD x[SABR_NUM_PARAMS];
        for (int p = 0; p < SABR_NUM_PARAMS; ++p) x[p] = params[p];
        for (auto& v : x) tape.registerInput(v);
        tape.newRecording();
        AD y = sabr_calibration_objective(x, data);
        tape.registerOutput(y);
        derivative(y) = 1.0;
        tape.computeAdjoints();
        if (it == 0) obj0 = value(y);
        sum_obj += value(y);
        for (int p = 0; p < SABR_NUM_PARAMS; ++p)
        {
            if (it == 0) g0[p] = derivative(x[p]);
            sum_g[p] += derivative(x[p]);
            params[p] += schedule[it].dp[p];
        }
    }
    printf("sabr0 %.17g", obj0);
    for (double v : g0) printf(" %.17g", v);
    printf("\nsabr_sum %.17g", sum_obj);
    for (double v : sum_g) printf(" %.17g", v);
    printf("\n");
}
