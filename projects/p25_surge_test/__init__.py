"""p25: surge and scale test harness — ramp / spike / soak / step scenarios against function, HTTP
or simulated targets, knee detection, Little's-law check, capacity headroom and a token-bucket
rate limiter that turns 5xx into 429 during an all-hands spike."""
