#include <UT/UT_Interrupt.h>

namespace interrupt {

// Interrupt checker.
struct InterruptChecker {
    UT_AutoInterrupt progress;

    InterruptChecker(const char * status_message)
        : progress(status_message) {
    }
};

bool check_interrupt(void *interrupt) {
    return !static_cast<InterruptChecker*>(interrupt)->progress.wasInterrupted();
}

} // namespace interrupt
