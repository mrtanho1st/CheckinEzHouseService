
import asyncio
import aiohttp
import argparse
import time
import statistics

async def send_request(session, url, image_bytes, semaphore):
    async with semaphore:
        start = time.perf_counter()

        try:
            form = aiohttp.FormData()
            form.add_field(
                "file",
                image_bytes,
                filename="author.jpg",
                content_type="image/jpeg"
            )

            async with session.post(url, data=form) as response:
                data = await response.json()
                latency = (time.perf_counter() - start) * 1000

                return response.status == 200 and "faces" in data, latency

        except Exception as e:
            print("Request failed:", e)
            return False, (time.perf_counter() - start) * 1000


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8085/detect")
    parser.add_argument("--image", default="author.jpg")
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    with open(args.image, "rb") as f:
        image_bytes = f.read()

    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        start = time.perf_counter()

        tasks = [
            send_request(session, args.url, image_bytes, semaphore)
            for _ in range(args.total)
        ]

        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

    successful = [lat for ok, lat in results if ok]
    failed = args.total - len(successful)

    print("\n========== LOAD TEST ==========")
    print(f"Total requests: {args.total}")
    print(f"Concurrency:    {args.concurrency}")
    print(f"Successful:     {len(successful)}")
    print(f"Failed:         {failed}")
    print(f"Total time:     {elapsed:.2f} s")
    print(f"Success RPS:    {len(successful)/elapsed:.2f}")

    if successful:
        successful.sort()

        def percentile(p):
            pos = (len(successful) - 1) * p / 100
            low = int(pos)
            high = min(low + 1, len(successful) - 1)
            return successful[low] + (successful[high] - successful[low]) * (pos - low)

        print(f"Avg latency:    {statistics.mean(successful):.2f} ms")
        print(f"P50 latency:    {percentile(50):.2f} ms")
        print(f"P95 latency:    {percentile(95):.2f} ms")
        print(f"P99 latency:    {percentile(99):.2f} ms")


if __name__ == "__main__":
    asyncio.run(main())
