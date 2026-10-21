// Inputs and XAD reference results for the ad-benchmarks Heston MC, SABR calibration and LIBOR
// swaption workloads, from that repository's own headers (src/heston.hpp, src/sabr.hpp,
// src/libor_swaption.hpp), so raads runs on identical inputs and can be checked against XAD.
//
// Writes target/adbench.bin (little-endian f64):
//   Heston: per path, z1[100] then z2[100] (10000 paths)
//   SABR:   per expiry F, T, beta, strikes[20], market_vols[20]; initial params[15];
//           perturbation schedule[500][15]
//   LIBOR:  per path, z[40] (10000 paths)
// and prints the XAD values: Heston price and gradient; SABR objective and gradient at the
// first iteration, and the sums over all 500 iterations; LIBOR price and gradient (delta,
// lambda[80], L0[80]).

#include "src/heston.hpp"
#include "src/libor_swaption.hpp"
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
    // the draws xad/libor_swaption_xad.cpp makes path by path (an even count per path, so
    // one generator over all paths gives the same numbers)
    auto market = defaultMarket();
    auto portfolio = defaultPortfolio();
    auto lz = libor_generate_all_samples(paths, market.lambda.size() / 2, 12354);
    for (auto& zp : lz)
        for (double z : zp) put(f, z);
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

    // LIBOR: pathwise adjoints, as in xad/libor_swaption_xad.cpp
    std::vector<AD> L, tmp1, tmp2, lambda, L0;
    double lprice = 0, lg[161] = {};
    for (int p = 0; p < paths; ++p)
    {
        tmp1.clear(); tmp2.clear(); L.clear();
        lambda.clear(); L0.clear();
        tape.clearAll();
        AD delta = market.delta;
        lambda.assign(market.lambda.begin(), market.lambda.end());
        L0.assign(market.L0.begin(), market.L0.end());
        tape.registerInput(delta);
        tape.registerInputs(lambda);
        tape.registerInputs(L0);
        tape.newRecording();
        L.assign(L0.begin(), L0.end());
        libor_path_gen(delta, L, lambda, lz[p]);
        AD v = libor_value_portfolio(delta, portfolio.maturities, portfolio.swaprates, L, tmp1, tmp2);
        tape.registerOutput(v);
        derivative(v) = 1.0;
        tape.computeAdjoints();
        lprice += value(v);
        lg[0] += derivative(delta);
        for (int i = 0; i < 80; ++i)
        {
            lg[1 + i] += derivative(lambda[i]);
            lg[81 + i] += derivative(L0[i]);
        }
    }
    printf("libor %.17g", lprice / paths);
    for (double v : lg) printf(" %.17g", v / paths);
    printf("\n");
}
