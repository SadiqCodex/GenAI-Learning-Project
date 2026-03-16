import asyncio
import time

async def function1():
    await asyncio.sleep(1)
    print("fun1")

async def function2():
    await asyncio.sleep(3)
    print("fun2")

async def function3():
    await asyncio.sleep(3)
    print("fun3")

async def main():
    start = time.perf_counter()

    # run all coroutines together
    await asyncio.gather(
        function1(),
        function2(),
        function3()
    )

    end = time.perf_counter()
    print(f"Total time (asyncio): {round(end - start, 2)} seconds")

asyncio.run(main())
