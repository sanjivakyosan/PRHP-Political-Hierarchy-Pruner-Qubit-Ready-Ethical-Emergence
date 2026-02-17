#!/usr/bin/env python3
"""
Enhanced PRHP Framework - Live Workflow Example

This script demonstrates the complete live workflow integrating all enhanced features:
- Live X/Twitter sentiment analysis
- WHO RSS feed monitoring
- Multi-qubit quantum simulation
- KPI dashboard publishing to IPFS
- Automated upkeep monitoring

Usage:
    python examples/live_workflow.py
    python examples/live_workflow.py --hashtag "#YourHashtag"
    python examples/live_workflow.py --no-quantum  # Use classical simulation
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prhp_enhanced import EnhancedPRHPFramework, run_live_workflow
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run Enhanced PRHP Live Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full live workflow
  python examples/live_workflow.py
  
  # Custom hashtag
  python examples/live_workflow.py --hashtag "#AIEthics"
  
  # Classical simulation (faster)
  python examples/live_workflow.py --no-quantum
  
  # Custom parameters
  python examples/live_workflow.py --levels 12 --monte 500
        """
    )
    
    parser.add_argument(
        '--hashtag',
        type=str,
        default='#AIEatsThePoor',
        help='X/Twitter hashtag to monitor (default: #AIEatsThePoor)'
    )
    
    parser.add_argument(
        '--sample-secs',
        type=int,
        default=30,
        help='Seconds to look back for tweets (default: 30)'
    )
    
    parser.add_argument(
        '--levels',
        type=int,
        default=18,
        help='Number of hierarchy levels (default: 18)'
    )
    
    parser.add_argument(
        '--monte',
        type=int,
        default=2000,
        help='Monte Carlo iterations (default: 2000, max: 5000)'
    )
    
    parser.add_argument(
        '--no-quantum',
        action='store_true',
        help='Use classical simulation instead of quantum'
    )
    
    parser.add_argument(
        '--no-multi-qubit',
        action='store_true',
        help='Disable multi-qubit W-state (use 2-qubit Bell state)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Run live workflow
    print("\n" + "="*70)
    print("Enhanced PRHP Framework - Live Workflow")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Hashtag: {args.hashtag}")
    print(f"  Sample window: {args.sample_secs} seconds")
    print(f"  Levels: {args.levels}")
    print(f"  Monte Carlo: {args.monte}")
    print(f"  Quantum: {not args.no_quantum}")
    print(f"  Multi-qubit: {not args.no_multi_qubit}")
    print(f"  Seed: {args.seed}")
    print()
    
    try:
        prhp, ipfs_cid = run_live_workflow(
            levels=args.levels,
            monte=args.monte,
            use_quantum=not args.no_quantum,
            multi_qubit=not args.no_multi_qubit,
            hashtag=args.hashtag,
            sample_secs=args.sample_secs,
            seed=args.seed
        )
        
        # Print summary
        print("\n" + "="*70)
        print("Workflow Summary")
        print("="*70)
        
        # KPI status
        print("\nKPI Status:")
        kpis = prhp.define_kpis()
        for variant, status in kpis.items():
            overall = status.get('overall', False)
            status_icon = "✓" if overall else "✗"
            print(f"  {status_icon} {variant}: {'PASS' if overall else 'FAIL'}")
        
        # Pruning efficacy
        print("\nPruning Efficacy:")
        try:
            efficacy = prhp.compute_pruning_efficacy(use_quantum=not args.no_quantum)
            for stressor, pct in efficacy.items():
                print(f"  {stressor}: {pct:.1f}%")
        except Exception as e:
            print(f"  ⚠ Could not compute: {e}")
        
        # IPFS CID
        if ipfs_cid:
            print(f"\n📊 KPI Dashboard: ipfs://{ipfs_cid}")
        
        print("\n" + "="*70)
        print("Workflow Complete")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

