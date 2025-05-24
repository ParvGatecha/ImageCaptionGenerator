import requests
import time
import concurrent.futures
import random
from datetime import datetime

# Configuration
FRONTEND_URL = "http://ai-tools"  # Your frontend service URL
NUM_REQUESTS = 100  # Total number of requests to make
CONCURRENT_USERS = 10  # Number of concurrent users
REQUEST_INTERVAL = 0.1  # Time between requests in seconds

def make_request():
    """Make a single request to the frontend"""
    try:
        start_time = time.time()
        response = requests.get(FRONTEND_URL)
        end_time = time.time()
        
        return {
            'status_code': response.status_code,
            'response_time': end_time - start_time,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
    except Exception as e:
        return {
            'status_code': 'ERROR',
            'error': str(e),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }

def worker():
    """Worker function for concurrent requests"""
    results = []
    for _ in range(NUM_REQUESTS // CONCURRENT_USERS):
        result = make_request()
        results.append(result)
        time.sleep(REQUEST_INTERVAL)
    return results

def main():
    print(f"Starting load test with {CONCURRENT_USERS} concurrent users")
    print(f"Total requests: {NUM_REQUESTS}")
    print(f"Target URL: {FRONTEND_URL}")
    print("-" * 50)

    all_results = []
    start_time = time.time()

    # Create a thread pool and execute requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
        futures = [executor.submit(worker) for _ in range(CONCURRENT_USERS)]
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate statistics
    successful_requests = sum(1 for r in all_results if r['status_code'] == 200)
    failed_requests = len(all_results) - successful_requests
    response_times = [r['response_time'] for r in all_results if 'response_time' in r]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0

    # Print results
    print("\nLoad Test Results:")
    print("-" * 50)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests: {failed_requests}")
    print(f"Average response time: {avg_response_time:.3f} seconds")
    print(f"Requests per second: {len(all_results) / total_time:.2f}")

    # Print some sample responses
    print("\nSample Responses:")
    print("-" * 50)
    for result in all_results[:5]:  # Show first 5 results
        print(f"Time: {result['timestamp']}, Status: {result['status_code']}, "
              f"Response Time: {result.get('response_time', 'N/A'):.3f}s")

if __name__ == "__main__":
    main() 