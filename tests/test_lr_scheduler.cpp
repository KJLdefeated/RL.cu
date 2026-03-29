// test_lr_scheduler.cpp
//
// Tests the LRScheduler class (constant + cosine with warmup).
// No GPU required.

#include <cstdio>
#include <cmath>
#include "training/lr_scheduler.h"

static int g_fails = 0;
#define PASS(name)      printf("[PASS] %s\n", (name))
#define FAIL(name, ...) do { \
    printf("[FAIL] %s: ", (name)); printf(__VA_ARGS__); printf("\n"); g_fails++; \
} while (0)

static bool near(float a, float b, float tol = 1e-6f) {
    return fabsf(a - b) < tol;
}

// =============================================================================
// Test 1: Constant schedule (no warmup)
// =============================================================================
static void test_constant_no_warmup() {
    const char* name = "Constant schedule (no warmup)";
    LRScheduler sched;
    sched.init_constant(1e-4f);

    bool ok = true;
    for (int s = 0; s < 100; s++) {
        float lr = sched.get_lr(s);
        if (!near(lr, 1e-4f)) {
            FAIL(name, "step %d: lr=%.8f, expected 1e-4", s, lr);
            ok = false;
            break;
        }
    }
    if (ok) PASS(name);
}

// =============================================================================
// Test 2: Constant schedule with warmup
// =============================================================================
static void test_constant_warmup() {
    const char* name = "Constant schedule with warmup";
    LRScheduler sched;
    sched.init_constant(1e-3f, 10);

    // step 0: lr = 1e-3 * 1/10 = 1e-4
    float lr0 = sched.get_lr(0);
    if (!near(lr0, 1e-4f)) {
        FAIL(name, "step 0: lr=%.8f, expected 1e-4", lr0);
        return;
    }

    // step 4: lr = 1e-3 * 5/10 = 5e-4
    float lr4 = sched.get_lr(4);
    if (!near(lr4, 5e-4f)) {
        FAIL(name, "step 4: lr=%.8f, expected 5e-4", lr4);
        return;
    }

    // step 9: lr = 1e-3 * 10/10 = 1e-3 (last warmup step)
    float lr9 = sched.get_lr(9);
    if (!near(lr9, 1e-3f)) {
        FAIL(name, "step 9: lr=%.8f, expected 1e-3", lr9);
        return;
    }

    // step 10: fully warmed up
    float lr10 = sched.get_lr(10);
    if (!near(lr10, 1e-3f)) {
        FAIL(name, "step 10: lr=%.8f, expected 1e-3", lr10);
        return;
    }

    PASS(name);
}

// =============================================================================
// Test 3: Cosine schedule (no warmup)
// =============================================================================
static void test_cosine_no_warmup() {
    const char* name = "Cosine schedule (no warmup)";
    LRScheduler sched;
    sched.init_cosine(1e-3f, 1e-5f, 0, 100);

    // step 0: should be base_lr
    float lr0 = sched.get_lr(0);
    if (!near(lr0, 1e-3f)) {
        FAIL(name, "step 0: lr=%.8f, expected 1e-3", lr0);
        return;
    }

    // step 50 (midpoint): lr = min + (max-min) * 0.5*(1+cos(pi*0.5)) = min + (max-min)*0.5
    float expected_mid = 1e-5f + (1e-3f - 1e-5f) * 0.5f;
    float lr50 = sched.get_lr(50);
    if (!near(lr50, expected_mid, 1e-5f)) {
        FAIL(name, "step 50: lr=%.8f, expected %.8f", lr50, expected_mid);
        return;
    }

    // step 99 (last): should be close to min_lr
    float lr99 = sched.get_lr(99);
    if (lr99 > 2e-5f) {
        FAIL(name, "step 99: lr=%.8f, expected near 1e-5", lr99);
        return;
    }

    // step 100 (beyond): should be exactly min_lr
    float lr100 = sched.get_lr(100);
    if (!near(lr100, 1e-5f)) {
        FAIL(name, "step 100: lr=%.8f, expected 1e-5", lr100);
        return;
    }

    PASS(name);
}

// =============================================================================
// Test 4: Cosine schedule with warmup
// =============================================================================
static void test_cosine_warmup() {
    const char* name = "Cosine schedule with warmup";
    LRScheduler sched;
    sched.init_cosine(1e-3f, 0.0f, 20, 120);

    // Warmup: steps 0..19, linear ramp from 0 to 1e-3
    float lr0 = sched.get_lr(0);
    float expected0 = 1e-3f * (1.0f / 20.0f);
    if (!near(lr0, expected0)) {
        FAIL(name, "step 0: lr=%.8f, expected %.8f", lr0, expected0);
        return;
    }

    float lr19 = sched.get_lr(19);
    float expected19 = 1e-3f * (20.0f / 20.0f);
    if (!near(lr19, expected19)) {
        FAIL(name, "step 19: lr=%.8f, expected %.8f", lr19, expected19);
        return;
    }

    // step 20: start of cosine decay, should be base_lr
    float lr20 = sched.get_lr(20);
    if (!near(lr20, 1e-3f)) {
        FAIL(name, "step 20: lr=%.8f, expected 1e-3", lr20);
        return;
    }

    // step 119 (last decay step): close to min_lr=0
    float lr119 = sched.get_lr(119);
    if (lr119 > 1e-5f) {
        FAIL(name, "step 119: lr=%.8f, expected near 0", lr119);
        return;
    }

    // Monotonically decreasing after warmup
    bool ok = true;
    for (int s = 21; s < 120; s++) {
        if (sched.get_lr(s) > sched.get_lr(s - 1) + 1e-8f) {
            FAIL(name, "not monotonically decreasing at step %d", s);
            ok = false;
            break;
        }
    }

    if (ok) PASS(name);
}

// =============================================================================
// Test 5: Warmup is monotonically increasing
// =============================================================================
static void test_warmup_monotonic() {
    const char* name = "Warmup is monotonically increasing";
    LRScheduler sched;
    sched.init_cosine(2e-4f, 1e-6f, 50, 200);

    bool ok = true;
    for (int s = 1; s < 50; s++) {
        if (sched.get_lr(s) <= sched.get_lr(s - 1)) {
            FAIL(name, "not increasing at step %d: %.8f <= %.8f",
                 s, sched.get_lr(s), sched.get_lr(s - 1));
            ok = false;
            break;
        }
    }
    if (ok) PASS(name);
}

// =============================================================================
// Test 6: Cosine endpoints
// =============================================================================
static void test_cosine_endpoints() {
    const char* name = "Cosine endpoints (start=base, end=min)";
    LRScheduler sched;
    sched.init_cosine(5e-4f, 1e-6f, 0, 1000);

    float lr_start = sched.get_lr(0);
    float lr_end = sched.get_lr(1000);

    if (!near(lr_start, 5e-4f)) {
        FAIL(name, "start: lr=%.8f, expected 5e-4", lr_start);
        return;
    }
    if (!near(lr_end, 1e-6f)) {
        FAIL(name, "end: lr=%.8f, expected 1e-6", lr_end);
        return;
    }

    PASS(name);
}

int main() {
    printf("=== LR Scheduler tests ===\n\n");

    test_constant_no_warmup();
    test_constant_warmup();
    test_cosine_no_warmup();
    test_cosine_warmup();
    test_warmup_monotonic();
    test_cosine_endpoints();

    printf("\n%s\n", g_fails ? "SOME TESTS FAILED" : "All tests PASSED.");
    return g_fails ? 1 : 0;
}
