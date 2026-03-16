import multiprocessing
import time
import os

# ---------- TOP LEVEL FUNCTIONS (IMPORTANT) ----------

def square(n):
    return n * n

def task(name):
    print(f"Task {name} running in Process ID:", os.getpid())

def worker(q, n):
    q.put(n * n)

# ---------------- EXAMPLE 1 ----------------
def example_pool():
    numbers = list(range(1, 10001))
    start = time.time()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        result = pool.map(square, numbers)

    end = time.time()

    print("\n[Example 1] Pool + map")
    print("Total numbers processed:", len(result))
    print("Time taken:", round(end - start, 2), "seconds")

# ---------------- EXAMPLE 2 ----------------
def example_process():
    print("\n[Example 2] Multiple Process")

    processes = []
    for i in range(5):
        p = multiprocessing.Process(target=task, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All processes completed")

# ---------------- EXAMPLE 3 ----------------
def example_queue():
    print("\n[Example 3] Queue Communication")

    q = multiprocessing.Queue()
    processes = []

    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(q, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    while not q.empty():
        print("Result:", q.get())

# ---------------- MAIN ENTRY ----------------
if __name__ == "__main__":
    multiprocessing.freeze_support()  # extra safe for Windows
    example_pool()
    example_process()
    example_queue()
