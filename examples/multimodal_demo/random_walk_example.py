#!/usr/bin/env python3
"""Random Walk Analysis Example for NaviGraph.

This script demonstrates the random walk functionality on a binary tree graph,
showing various use cases including backtracking comparison, performance analysis,
and statistical evaluation.

Example for binary tree with height 7 (127 nodes total):
- Root node: 0
- Leaf nodes: 70-127
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import NaviGraph
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))  # Add project root to path
from navigraph.core.graph import GraphStructure


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def example_1_basic_walks():
    """Example 1: Basic fixed-length random walks."""
    print_section("Example 1: Basic Fixed-Length Random Walks")

    # Create binary tree (height 7 = 127 nodes)
    graph = GraphStructure.from_config('binary_tree', {'height': 7})
    print(f"Created binary tree: {graph.num_nodes} nodes, {graph.num_edges} edges")

    # Generate 10 walks of length 5 from root
    print("\nGenerating 10 random walks of 5 steps from root node...")
    paths = graph.random_walks(
        start_node=0,  # Root
        n_walks=10,
        max_steps=5,
        seed=42  # Reproducible
    )

    print(f"Generated {len(paths)} walks")
    print("\nFirst 3 walks:")
    for i, path in enumerate(paths[:3]):
        print(f"  Walk {i+1}: {path} (length: {len(path)-1} steps)")


def example_2_target_directed():
    """Example 2: Target-directed walks with success rate."""
    print_section("Example 2: Target-Directed Walks")

    graph = GraphStructure.from_config('binary_tree', {'height': 7})

    # Walk from root to a leaf node
    target = 127  # Rightmost leaf
    print(f"Walking from root (0) to leaf node ({target})")
    print("Generating 1000 walks with max 50 steps...")

    paths, stats = graph.random_walks(
        start_node=0,
        target_node=target,
        n_walks=1000,
        max_steps=50,
        backtrack_prob=0.0,
        return_stats=True,
        seed=42
    )

    print(f"\nResults:")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Average path length: {stats['mean_length']:.2f} ± {stats['std_length']:.2f} steps")
    print(f"  Min path length: {stats['min_length']} steps")
    print(f"  Max path length: {stats['max_length']} steps")
    print(f"  Median path length: {stats['median_length']:.1f} steps")


def example_3_backtracking_comparison():
    """Example 3: Compare different backtracking probabilities."""
    print_section("Example 3: Backtracking Comparison")

    graph = GraphStructure.from_config('binary_tree', {'height': 7})

    # Test different backtracking probabilities
    backtrack_probs = [0.0, 0.3, 0.5, 0.8]
    n_walks = 1000
    max_steps = 50
    target = 127  # Rightmost leaf

    print(f"Comparing backtracking probabilities from root (0) to leaf ({target})")
    print(f"Running {n_walks} walks with max {max_steps} steps each...\n")

    results = []
    for prob in backtrack_probs:
        paths, stats = graph.random_walks(
            start_node=0,
            target_node=target,
            n_walks=n_walks,
            max_steps=max_steps,
            backtrack_prob=prob,
            return_stats=True,
            seed=42
        )
        results.append(stats)

        print(f"Backtrack probability = {prob:.1f}:")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Average length: {stats['mean_length']:.2f} steps")
        print(f"  Std deviation: {stats['std_length']:.2f} steps")
        print()

    # Plot comparison
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot success rates
        success_rates = [r['success_rate'] for r in results]
        ax1.bar(range(len(backtrack_probs)), success_rates)
        ax1.set_xticks(range(len(backtrack_probs)))
        ax1.set_xticklabels([f'{p:.1f}' for p in backtrack_probs])
        ax1.set_xlabel('Backtrack Probability')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate vs Backtracking')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)

        # Plot average path lengths
        mean_lengths = [r['mean_length'] for r in results]
        std_lengths = [r['std_length'] for r in results]
        ax2.errorbar(backtrack_probs, mean_lengths, yerr=std_lengths,
                     marker='o', capsize=5, linewidth=2, markersize=8)
        ax2.set_xlabel('Backtrack Probability')
        ax2.set_ylabel('Average Path Length (steps)')
        ax2.set_title('Path Length vs Backtracking')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        output_path = Path(__file__).parent / 'random_walk_backtracking.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
        plt.close()

    except Exception as e:
        print(f"Could not generate plot: {e}")


def example_4_performance_comparison():
    """Example 4: Serial vs parallel performance."""
    print_section("Example 4: Performance Comparison (Serial vs Parallel)")

    graph = GraphStructure.from_config('binary_tree', {'height': 7})

    # Test different numbers of walks
    walk_counts = [100, 1000, 5000, 10000]
    max_steps = 30

    print(f"Comparing serial (n_jobs=1) vs parallel (n_jobs=-1) execution")
    print(f"Each walk: max {max_steps} steps from root\n")

    print(f"{'N Walks':<12} {'Serial (s)':<15} {'Parallel (s)':<15} {'Speedup':<12}")
    print("-" * 60)

    for n_walks in walk_counts:
        # Serial execution
        start = time.time()
        paths_serial = graph.random_walks(
            start_node=0,
            n_walks=n_walks,
            max_steps=max_steps,
            n_jobs=1,
            seed=42
        )
        serial_time = time.time() - start

        # Parallel execution
        start = time.time()
        paths_parallel = graph.random_walks(
            start_node=0,
            n_walks=n_walks,
            max_steps=max_steps,
            n_jobs=-1,  # Use all cores
            seed=42
        )
        parallel_time = time.time() - start

        speedup = serial_time / parallel_time if parallel_time > 0 else 0

        print(f"{n_walks:<12} {serial_time:<15.3f} {parallel_time:<15.3f} {speedup:<12.2f}x")

        # Verify results are identical (same seed should give same results)
        if paths_serial == paths_parallel:
            print(f"  ✓ Serial and parallel results identical")
        else:
            print(f"  ⚠ Warning: Serial and parallel results differ!")

    print("\nNote: Speedup depends on number of CPU cores and system load")
    print("For small n_walks, serial may be faster due to multiprocessing overhead")


def example_5_path_length_distribution():
    """Example 5: Analyze path length distribution."""
    print_section("Example 5: Path Length Distribution Analysis")

    graph = GraphStructure.from_config('binary_tree', {'height': 7})

    # Generate many walks
    n_walks = 5000
    target = 127  # Rightmost leaf

    print(f"Analyzing path length distribution")
    print(f"From root (0) to leaf ({target})")
    print(f"Generating {n_walks} walks...\n")

    paths, stats = graph.random_walks(
        start_node=0,
        target_node=target,
        n_walks=n_walks,
        max_steps=100,
        backtrack_prob=0.0,
        return_stats=True,
        seed=42
    )

    # Extract path lengths
    path_lengths = [len(path) - 1 for path in paths]
    successful_paths = [path for path in paths if path[-1] == target]
    successful_lengths = [len(path) - 1 for path in successful_paths]

    print(f"Statistics:")
    print(f"  Total walks: {len(paths)}")
    print(f"  Successful: {len(successful_paths)} ({stats['success_rate']:.1%})")
    print(f"  Mean length (all): {stats['mean_length']:.2f} steps")
    print(f"  Mean length (successful): {np.mean(successful_lengths):.2f} steps")
    print(f"  Median length: {stats['median_length']:.1f} steps")
    print(f"  Min length: {stats['min_length']} steps")
    print(f"  Max length: {stats['max_length']} steps")

    # For comparison, calculate shortest path
    shortest_path = graph.get_shortest_path(0, target)
    print(f"\n  Shortest path length: {len(shortest_path) - 1} steps")
    print(f"  Random walk efficiency: {(len(shortest_path)-1) / stats['mean_length']:.1%}")

    # Plot distribution
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot all path lengths
        ax1.hist(path_lengths, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(stats['mean_length'], color='red', linestyle='--',
                    linewidth=2, label=f"Mean: {stats['mean_length']:.1f}")
        ax1.axvline(stats['median_length'], color='green', linestyle='--',
                    linewidth=2, label=f"Median: {stats['median_length']:.1f}")
        ax1.set_xlabel('Path Length (steps)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Path Length Distribution (All {n_walks} Walks)')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Plot successful path lengths only
        ax2.hist(successful_lengths, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax2.axvline(np.mean(successful_lengths), color='red', linestyle='--',
                    linewidth=2, label=f"Mean: {np.mean(successful_lengths):.1f}")
        ax2.axvline(len(shortest_path)-1, color='blue', linestyle='--',
                    linewidth=2, label=f"Shortest: {len(shortest_path)-1}")
        ax2.set_xlabel('Path Length (steps)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Successful Walks Only ({len(successful_paths)} walks)')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = Path(__file__).parent / 'random_walk_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to: {output_path}")
        plt.close()

    except Exception as e:
        print(f"\nCould not generate plot: {e}")


def example_6_reproducibility():
    """Example 6: Demonstrate reproducibility with seeds."""
    print_section("Example 6: Reproducibility with Seeds")

    graph = GraphStructure.from_config('binary_tree', {'height': 7})

    print("Demonstrating reproducibility with seeds...\n")

    # Run same walks twice with same seed
    seed = 12345
    paths1 = graph.random_walks(start_node=0, n_walks=5, max_steps=10, seed=seed)
    paths2 = graph.random_walks(start_node=0, n_walks=5, max_steps=10, seed=seed)

    print("With same seed (12345):")
    print(f"  Run 1 first path: {paths1[0]}")
    print(f"  Run 2 first path: {paths2[0]}")
    print(f"  Identical: {paths1 == paths2} ✓")

    # Run with different seed
    paths3 = graph.random_walks(start_node=0, n_walks=5, max_steps=10, seed=99999)

    print(f"\nWith different seed (99999):")
    print(f"  Run 3 first path: {paths3[0]}")
    print(f"  Different from run 1: {paths1[0] != paths3[0]} ✓")

    # Run without seed (non-deterministic)
    paths4 = graph.random_walks(start_node=0, n_walks=5, max_steps=10, seed=None)
    paths5 = graph.random_walks(start_node=0, n_walks=5, max_steps=10, seed=None)

    print(f"\nWithout seed (random):")
    print(f"  Run 4 first path: {paths4[0]}")
    print(f"  Run 5 first path: {paths5[0]}")
    print(f"  Likely different: {paths4[0] != paths5[0]} (may occasionally be same)")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("  NaviGraph Random Walk Examples")
    print("  Binary Tree Graph (Height 7 = 127 nodes)")
    print("="*70)

    try:
        # Run all examples
        example_1_basic_walks()
        example_2_target_directed()
        example_3_backtracking_comparison()
        example_4_performance_comparison()
        example_5_path_length_distribution()
        example_6_reproducibility()

        print("\n" + "="*70)
        print("  All examples completed successfully!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
