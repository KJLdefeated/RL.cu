// CPU-only tests for the constant and cosine schedules, including warmup.
#include <gtest/gtest.h>
#include "training/lr_scheduler.h"

TEST(LRSchedulerTest, ConstantWithoutWarmup) {
    LRScheduler sched;
    sched.init_constant(1e-4f);
    for (int step = 0; step < 100; ++step) {
        EXPECT_FLOAT_EQ(sched.get_lr(step), 1e-4f) << "step=" << step;
    }
}

TEST(LRSchedulerTest, ConstantWithWarmup) {
    LRScheduler sched;
    sched.init_constant(1e-3f, 10);
    EXPECT_NEAR(sched.get_lr(0), 1e-4f, 1e-6f);
    EXPECT_NEAR(sched.get_lr(4), 5e-4f, 1e-6f);
    EXPECT_NEAR(sched.get_lr(9), 1e-3f, 1e-6f);
    EXPECT_NEAR(sched.get_lr(10), 1e-3f, 1e-6f);
    EXPECT_NEAR(sched.get_lr(100), 1e-3f, 1e-6f);
}

TEST(LRSchedulerTest, CosineWithoutWarmup) {
    LRScheduler sched;
    sched.init_cosine(1e-3f, 1e-5f, 0, 100);
    EXPECT_NEAR(sched.get_lr(0), 1e-3f, 1e-6f);
    EXPECT_NEAR(sched.get_lr(50), 1e-5f + (1e-3f - 1e-5f) * 0.5f, 1e-5f);
    EXPECT_LT(sched.get_lr(99), 2e-5f);
    EXPECT_NEAR(sched.get_lr(100), 1e-5f, 1e-6f);
}

TEST(LRSchedulerTest, CosineWithWarmup) {
    LRScheduler sched;
    sched.init_cosine(1e-3f, 0.0f, 20, 120);
    EXPECT_NEAR(sched.get_lr(0), 1e-3f / 20, 1e-6f);
    EXPECT_NEAR(sched.get_lr(19), 1e-3f, 1e-6f);
    EXPECT_NEAR(sched.get_lr(20), 1e-3f, 1e-6f);
    EXPECT_LT(sched.get_lr(119), 1e-5f);
    for (int step = 21; step < 120; ++step) {
        EXPECT_LE(sched.get_lr(step), sched.get_lr(step - 1) + 1e-8f)
            << "step=" << step;
    }
}

TEST(LRSchedulerTest, WarmupIsMonotonicallyIncreasing) {
    LRScheduler sched;
    sched.init_cosine(2e-4f, 1e-6f, 50, 200);
    for (int step = 1; step < 50; ++step) {
        EXPECT_GT(sched.get_lr(step), sched.get_lr(step - 1)) << "step=" << step;
    }
}

TEST(LRSchedulerTest, CosineEndpointsAndStepsBeyondTraining) {
    LRScheduler sched;
    sched.init_cosine(5e-4f, 1e-6f, 0, 1000);
    EXPECT_NEAR(sched.get_lr(0), 5e-4f, 1e-6f);
    EXPECT_NEAR(sched.get_lr(1000), 1e-6f, 1e-6f);
    EXPECT_NEAR(sched.get_lr(2000), 1e-6f, 1e-6f);
}

TEST(LRSchedulerTest, NoDecayStepsKeepsBaseRateAfterWarmup) {
    LRScheduler sched;
    sched.init_cosine(1e-3f, 0.0f, 10, 10);
    EXPECT_NEAR(sched.get_lr(9), 1e-3f, 1e-6f);
    EXPECT_NEAR(sched.get_lr(10), 1e-3f, 1e-6f);
    EXPECT_NEAR(sched.get_lr(20), 1e-3f, 1e-6f);
}
