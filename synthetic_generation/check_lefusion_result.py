"""
check_lefusion_result.py

Checks on a LeFusion job that was started with modal_lefusion.py's spawn-based
entrypoint. Since the job runs detached from your local connection, you can
run this from a totally different network/session than the one that started
it -- useful if your connection dropped mid-run (e.g. travel wifi), since the
job on Modal's side kept running regardless.

Usage:
    python check_lefusion_result.py <call_id>

The call ID is printed when you run modal_lefusion.py, and is also visible
in the Modal web dashboard for this app under "Function Calls".
"""

import sys
import modal

if len(sys.argv) < 2:
    print("Usage: python check_lefusion_result.py <call_id>")
    print("(the call ID was printed when you ran modal_lefusion.py, or check "
          "the Modal web dashboard under Function Calls)")
    sys.exit(1)

call_id = sys.argv[1]
call = modal.FunctionCall.from_id(call_id)

print(f"Checking call {call_id}...")
try:
    result = call.get(timeout=5)
    print("\nJob finished. Result:")
    print(result)
except TimeoutError:
    print("\nStill running. Run this same command again later to check again.")
except Exception as e:
    print(f"\nJob finished with an error.")
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception repr: {repr(e)}")
    print(f"Exception args: {e.args}")
    print("\nIf this is still unclear, check the Modal web dashboard for this "
          "call's Logs tab -- the real traceback was printed live during the "
          "run and should be visible there.")
