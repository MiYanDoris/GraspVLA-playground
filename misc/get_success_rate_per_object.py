import os
import re
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Calculate success rate per object category from video files')
    parser.add_argument('--result_path', type=str, default='data/playground_test/videos',
                       help='Path to the directory containing video files')
    
    args = parser.parse_args()
    result_path = args.result_path
    
    videos = os.listdir(result_path)

    # Dictionary to store counts for each category
    category_stats = defaultdict(lambda: {'success': 0, 'fail': 0, 'total': 0})

    # Parse each video filename
    for video in videos:
        if not video.endswith('.mp4'):
            continue
        
        # Remove .mp4 extension
        name_without_ext = video[:-4]
        
        # Check if it has success/fail indicator
        if name_without_ext.endswith('_success'):
            result = 'success'
            name_parts = name_without_ext[:-8]  # Remove '_success'
        elif name_without_ext.endswith('_fail'):
            result = 'fail'
            name_parts = name_without_ext[:-5]  # Remove '_fail'
        else:
            # Files without explicit success/fail - skip or treat as unknown
            print(f"Skipping file without success/fail indicator: {video}")
            continue
        
        # Remove '_x264' if present
        if name_parts.endswith('_x264'):
            name_parts = name_parts[:-6]
        
        # Extract category (first part before first underscore)
        category = name_parts.split('_')[4]
        
        # Update statistics
        category_stats[category]['total'] += 1
        if result == 'success':
            category_stats[category]['success'] += 1
        else:
            category_stats[category]['fail'] += 1

    # Calculate and display success rates
    print("Success Rate per Category:")
    print("=" * 50)
    print(f"{'Category':<20} {'Success':<8} {'Fail':<6} {'Total':<7} {'Success Rate':<12}")
    print("-" * 50)

    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"{category:<20} {stats['success']:<8} {stats['fail']:<6} {stats['total']:<7} {success_rate:>8.1f}%")

    # Overall statistics
    total_videos = sum(stats['total'] for stats in category_stats.values())
    total_success = sum(stats['success'] for stats in category_stats.values())
    overall_success_rate = (total_success / total_videos) * 100 if total_videos > 0 else 0

    print("-" * 50)
    print(f"{'OVERALL':<20} {total_success:<8} {total_videos-total_success:<6} {total_videos:<7} {overall_success_rate:>8.1f}%")
    print(f"\nTotal categories: {len(category_stats)}")

if __name__ == "__main__":
    main()

