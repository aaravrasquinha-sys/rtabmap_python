"""
Run the full synthetic gate suite (no hardware / camera required):

    python -m pyslam.selftest

Every module in this codebase should be considered untrustworthy until
this passes. Run it immediately after `git pull` / unzipping, before
touching the camera.
"""
import sys
import time
import traceback

sys.path.insert(0, ".")

from tests.gates.test_g0 import ALL_TESTS


def main() -> int:
    print(f"pySLAM Phase 0 self-test -- {len(ALL_TESTS)} checks\n")
    n_pass, n_fail = 0, 0
    t_start = time.time()
    for test in ALL_TESTS:
        name = test.__name__
        t0 = time.time()
        try:
            test()
            n_pass += 1
        except AssertionError as e:
            n_fail += 1
            print(f"  [FAIL] {name}: {e}")
        except Exception:
            n_fail += 1
            print(f"  [ERROR] {name}:")
            traceback.print_exc()
        finally:
            dt = time.time() - t0
            if dt > 1.0:
                print(f"      ({dt:.1f}s)")

    total = time.time() - t_start
    print(f"\n{n_pass}/{len(ALL_TESTS)} passed in {total:.1f}s")
    if n_fail > 0:
        print("SELFTEST FAILED -- do not proceed to hardware until this is green.")
        return 1
    print("SELFTEST PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
