#!/usr/bin/env python3
"""
Run Constant Proportion Experiments

This script runs the constant proportion synthetic loop experiments across
multiple fixed proportions to determine the critical threshold for MADness.

Experiments:
- p=0.0:  100% real (baseline, no synthetic)
- p=0.25: 25% synthetic, 75% real
- p=0.5:  50% synthetic, 50% real
- p=0.75: 75% synthetic, 25% real
- p=0.9:  90% synthetic, 10% real
- p=1.0:  100% synthetic (full synthetic loop)

Each experiment maintains a CONSTANT training set size of 60k samples.
"""

import subprocess
import os
import sys
import time
import argparse

# Proportions to test
PROPORTIONS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]

# Default settings
DEFAULT_GENERATIONS = 9
DEFAULT_DATA_PATH = '../data'


def run_experiment(proportion, generations, data_path, dry_run=False):
    """Run a single constant proportion experiment."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, 'main_constant_proportion.py')

    cmd = [
        sys.executable,
        script_path,
        '--proportion', str(proportion),
        '--generations', str(generations),
        '--data_path', data_path
    ]

    print("\n" + "="*70)
    print(f"EXPERIMENT: p={proportion} ({proportion*100:.0f}% synthetic)")
    print("="*70)
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN] Skipping execution")
        return True

    print()
    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ Experiment p={proportion} completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment p={proportion} FAILED with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n✗ Experiment p={proportion} interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run constant proportion synthetic loop experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python run_constant_proportion_experiments.py

  # Run specific proportions only
  python run_constant_proportion_experiments.py --proportions 0.5 0.75 0.9

  # Dry run to see what would be executed
  python run_constant_proportion_experiments.py --dry-run

  # Run with fewer generations (faster)
  python run_constant_proportion_experiments.py --generations 5
"""
    )

    parser.add_argument('--proportions', type=float, nargs='+', default=PROPORTIONS,
                        help=f'Proportions to test (default: {PROPORTIONS})')
    parser.add_argument('--generations', type=int, default=DEFAULT_GENERATIONS,
                        help=f'Number of generations per experiment (default: {DEFAULT_GENERATIONS})')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH,
                        help=f'Path to MNIST data (default: {DEFAULT_DATA_PATH})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue with next experiment if one fails')

    args = parser.parse_args()

    # Validate proportions
    for p in args.proportions:
        if not 0.0 <= p <= 1.0:
            print(f"Error: Proportion {p} must be between 0.0 and 1.0")
            sys.exit(1)

    # Sort proportions for consistent ordering
    proportions = sorted(args.proportions)

    print("="*70)
    print("CONSTANT PROPORTION SYNTHETIC LOOP EXPERIMENTS")
    print("="*70)
    print(f"Proportions to test: {proportions}")
    print(f"Generations per experiment: {args.generations}")
    print(f"Total training samples: 60,000 (constant)")
    print(f"Data path: {args.data_path}")
    print()

    print("Experiment Overview:")
    print("-"*70)
    print(f"{'Proportion':<12} {'Real Samples':<15} {'Synthetic Samples':<18} {'Output Dir':<25}")
    print("-"*70)
    for p in proportions:
        n_syn = int(60000 * p)
        n_real = 60000 - n_syn
        out_dir = f"./data/gan_outputs_p{int(p*100)}/"
        print(f"{p:<12.2f} {n_real:<15} {n_syn:<18} {out_dir:<25}")
    print("-"*70)
    print()

    if args.dry_run:
        print("[DRY RUN MODE - No experiments will be executed]")

    # Run experiments
    total_start = time.time()
    results = {}

    for i, proportion in enumerate(proportions):
        print(f"\n[{i+1}/{len(proportions)}] Starting experiment p={proportion}")

        success = run_experiment(
            proportion=proportion,
            generations=args.generations,
            data_path=args.data_path,
            dry_run=args.dry_run
        )

        results[proportion] = success

        if not success and not args.continue_on_error and not args.dry_run:
            print("\nStopping due to experiment failure. Use --continue-on-error to continue.")
            break

    # Summary
    total_elapsed = time.time() - total_start

    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    if not args.dry_run:
        print(f"{'Proportion':<12} {'Status':<15} {'Output Dir':<30}")
        print("-"*70)
        for p in proportions:
            status = "✓ SUCCESS" if results.get(p, False) else "✗ FAILED/SKIPPED"
            out_dir = f"./data/gan_outputs_p{int(p*100)}/"
            print(f"{p:<12.2f} {status:<15} {out_dir:<30}")
        print("-"*70)

        successful = sum(1 for v in results.values() if v)
        print(f"\nCompleted: {successful}/{len(proportions)} experiments")
        print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    else:
        print("[DRY RUN] No experiments were executed")

    print("="*70)

    # Suggest next steps
    if not args.dry_run and successful > 0:
        print("\nNext Steps:")
        print("1. Compute FID scores:")
        print("   python compute_fid_constant_proportion.py")
        print("2. Run Jacobian analysis:")
        print("   python jacobian_constant_proportion.py")
        print("3. Create comparison plots:")
        print("   python compare_proportions.py")


if __name__ == "__main__":
    main()
