#pragma once
#include <cmath>
#include <algorithm>

enum class LRScheduleType {
    Constant,
    Cosine,
};

class LRScheduler {
public:
    LRScheduler() = default;

    void init_constant(float base_lr, int warmup_steps = 0) {
        type_         = LRScheduleType::Constant;
        base_lr_      = base_lr;
        min_lr_       = base_lr;
        warmup_steps_ = warmup_steps;
        total_steps_  = 0;  // unused for constant
    }

    void init_cosine(float base_lr, float min_lr, int warmup_steps, int total_steps) {
        type_         = LRScheduleType::Cosine;
        base_lr_      = base_lr;
        min_lr_       = min_lr;
        warmup_steps_ = warmup_steps;
        total_steps_  = total_steps;
    }

    float get_lr(int step) const {
        // Warmup phase: linear ramp from 0 to base_lr
        if (warmup_steps_ > 0 && step < warmup_steps_) {
            return base_lr_ * ((float)(step + 1) / (float)warmup_steps_);
        }

        if (type_ == LRScheduleType::Constant) {
            return base_lr_;
        }

        // Cosine decay phase
        int decay_steps = total_steps_ - warmup_steps_;
        if (decay_steps <= 0) return base_lr_;

        int progress = step - warmup_steps_;
        if (progress >= decay_steps) return min_lr_;

        float cosine = 0.5f * (1.0f + cosf((float)M_PI * (float)progress / (float)decay_steps));
        return min_lr_ + (base_lr_ - min_lr_) * cosine;
    }

private:
    LRScheduleType type_         = LRScheduleType::Constant;
    float          base_lr_      = 1e-5f;
    float          min_lr_       = 0.0f;
    int            warmup_steps_ = 0;
    int            total_steps_  = 0;
};
