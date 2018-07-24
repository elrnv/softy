#include <UT/UT_Interrupt.h>

namespace interrupt {

// Interrupt checker.
struct InterruptChecker {
    UT_AutoInterrupt progress;

    InterruptChecker()
        : progress("Solving Softy") {
    }
};

bool check_interrupt(void *interrupt) {
    return !static_cast<InterruptChecker*>(interrupt)->progress.wasInterrupted();
}

} // namespace interrupt
