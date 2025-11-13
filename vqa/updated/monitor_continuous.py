#!/usr/bin/env python3
"""
Continuously monitor metrics at fixed intervals
"""
import time
import subprocess
from datetime import datetime

CHECK_INTERVAL = 60  # seconds
MAX_CHECKS = 180  # 3 hours worth of checks

def run_monitor():
    """Run the monitoring script and capture output"""
    result = subprocess.run(
        ['python', 'monitor_metrics.py'],
        capture_output=True,
        text=True
    )
    return result.stdout

def main():
    print("🔄 Starting continuous monitoring...")
    print(f"   Check interval: {CHECK_INTERVAL} seconds")
    print(f"   Max duration: {MAX_CHECKS * CHECK_INTERVAL / 60:.0f} minutes")
    print("=" * 80)

    check_count = 0
    last_match_count = 0

    while check_count < MAX_CHECKS:
        check_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n🕐 Check #{check_count} at {timestamp}")
        print("-" * 80)

        output = run_monitor()
        print(output)

        # Extract match count from output
        for line in output.split('\n'):
            if 'Total matches found:' in line:
                current_matches = int(line.split(':')[1].strip())
                if current_matches > last_match_count:
                    print(f"\n📈 NEW ENTRIES: {current_matches - last_match_count} new matches!")
                    last_match_count = current_matches
                break

        # Check if we're done (90 entries expected)
        if last_match_count >= 90:
            print("\n✅ All 90 entries completed!")
            break

        print(f"\n⏳ Waiting {CHECK_INTERVAL} seconds until next check...")
        time.sleep(CHECK_INTERVAL)

    print("\n🏁 Monitoring complete!")

if __name__ == "__main__":
    main()
