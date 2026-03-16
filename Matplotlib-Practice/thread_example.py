import threading
import time
from concurrent.futures import ThreadPoolExecutor

def fun(seconds):
    print(f"sleeping for {seconds} seconds")
    time.sleep(seconds)
    print(f"done sleeping {seconds}")

# ---------------- NORMAL CODE ----------------
print("\n--- Normal Execution ---")
start = time.perf_counter()

fun(4)
fun(2)
fun(1)

end = time.perf_counter()
print(f"Total time (Normal): {round(end - start, 2)} seconds")


# ---------------- MANUAL THREADING ----------------
print("\n--- Manual Threading ---")
start = time.perf_counter()

t1 = threading.Thread(target=fun, args=(4,))
t2 = threading.Thread(target=fun, args=(2,))
t3 = threading.Thread(target=fun, args=(1,))

t1.start()
t2.start()
t3.start()

t1.join()
t2.join()
t3.join()

end = time.perf_counter()
print(f"Total time (Threading): {round(end - start, 2)} seconds")


# ---------------- THREAD POOL EXECUTOR ----------------
print("\n--- ThreadPoolExecutor ---")
start = time.perf_counter()

with ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(fun, [4, 2, 1])

end = time.perf_counter()
print(f"Total time (ThreadPoolExecutor): {round(end - start, 2)} seconds")

print("\nAll executions finished")
