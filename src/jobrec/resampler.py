# Standard library imports
from collections import Counter
from typing import Dict, List, Optional, Set

# Third-party library imports
import pandas as pd
import numpy as np

def balance_resume_dataset(df: pd.DataFrame, samples_per_category: int = 30) -> pd.DataFrame:
    """Balance resume dataset by sampling equal numbers from each category."""
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    print("\n" + "="*60)
    print("BALANCING RESUME DATASET")
    print("="*60)
    
    balanced_dfs = []
    category_stats = []
    
    for category in sorted(df['category'].unique()):
        category_df = df[df['category'] == category]
        available = len(category_df)
        
        if available >= samples_per_category:
            # Sample exactly samples_per_category
            sampled = category_df.sample(n=samples_per_category, random_state=42)
            selected = samples_per_category
        else:
            # If not enough samples, take all and note the shortage
            sampled = category_df
            selected = available
            print(f"  Warning: {category} has only {available} samples (needed {samples_per_category})")
        
        balanced_dfs.append(sampled)
        category_stats.append({
            'category': category,
            'available': available,
            'selected': selected,
            'shortage': max(0, samples_per_category - available)
        })
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Display summary
    print(f"\nCategory distribution:")
    print(f"  {'Category':25s} {'Available':>10s} {'Selected':>10s} {'Shortage':>10s}")
    print("  " + "-"*58)
    
    for stat in category_stats:
        shortage_str = str(stat['shortage']) if stat['shortage'] > 0 else '-'
        print(f"  {stat['category']:25s} {stat['available']:10,d} {stat['selected']:10,d} {shortage_str:>10s}")
    
    print(f"\nTotal samples: {len(balanced_df):,}")
    print("="*60 + "\n")
    
    return balanced_df

def optimized_job_sampler(
    df: pd.DataFrame,
    domain_col: str,
    thresholds: Dict[str, int],
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Optimized job sampling using vectorized operations and efficient data structures.
    
    This is much faster than the iterative approach but may be slightly less optimal
    in terms of domain distribution.
    """
    np.random.seed(random_seed)
    
    print("  Using optimized vectorized sampler...")
    
    # Build job-domain mapping efficiently
    job_to_domains = {}
    domain_to_jobs = {domain: [] for domain in thresholds.keys()}
    
    for idx, row in df.iterrows():
        domains = row[domain_col]
        if isinstance(domains, str):
            domains = [domains]
        elif not isinstance(domains, list):
            continue
        
        # Only consider relevant domains
        relevant = [d for d in domains if d in thresholds]
        if relevant:
            job_to_domains[idx] = relevant
            for domain in relevant:
                domain_to_jobs[domain].append(idx)
    
    # Calculate domain flexibility (inverse for scoring)
    job_flexibility = {
        job_id: 1.0 / len(domains) 
        for job_id, domains in job_to_domains.items()
    }
    
    # Track selections
    selected_jobs = set()
    domain_counts = {domain: 0 for domain in thresholds.keys()}
    assignments = []
    
    # Priority queue approach: process domains by scarcity
    domain_scarcity = {
        domain: thresholds[domain] / max(1, len(domain_to_jobs[domain]))
        for domain in thresholds.keys()
    }
    
    # Sort domains by scarcity (highest first - these are hardest to fill)
    sorted_domains = sorted(domain_scarcity.items(), key=lambda x: x[1], reverse=True)
    
    # Phase 1: Fill scarce domains first
    for domain, _ in sorted_domains:
        target = thresholds[domain]
        current = domain_counts[domain]
        needed = target - current
        
        if needed <= 0:
            continue
        
        # Get available jobs for this domain
        available_jobs = [
            job for job in domain_to_jobs[domain] 
            if job not in selected_jobs
        ]
        
        if not available_jobs:
            continue
        
        # Score jobs by flexibility (prefer less flexible jobs)
        job_scores = [
            (job, job_flexibility[job]) 
            for job in available_jobs
        ]
        job_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top jobs
        for job, _ in job_scores[:needed]:
            selected_jobs.add(job)
            domain_counts[domain] += 1
            assignments.append({
                'job_id': job,
                'assigned_domain': domain
            })
            
            # Update counts for all domains this job belongs to
            for other_domain in job_to_domains[job]:
                if other_domain != domain and other_domain in domain_counts:
                    domain_counts[other_domain] = min(
                        domain_counts[other_domain] + 1,
                        thresholds[other_domain]
                    )
    
    # Phase 2: Fill remaining gaps with any available jobs
    for domain in thresholds.keys():
        target = thresholds[domain]
        current = domain_counts[domain]
        needed = target - current
        
        if needed <= 0:
            continue
        
        # Get any remaining jobs for this domain
        available_jobs = [
            job for job in domain_to_jobs[domain] 
            if job not in selected_jobs
        ]
        
        # Randomly sample to fill gaps
        if available_jobs:
            sample_size = min(needed, len(available_jobs))
            sampled = np.random.choice(available_jobs, sample_size, replace=False)
            
            for job in sampled:
                selected_jobs.add(job)
                domain_counts[domain] += 1
                assignments.append({
                    'job_id': job,
                    'assigned_domain': domain
                })
    
    # Create result DataFrame
    if assignments:
        result_df = pd.DataFrame(assignments)
        
        # Add original domains
        orig_col = 'job_domain_original' if 'job_domain_original' in df.columns else domain_col
        result_df['original_domains'] = result_df['job_id'].map(
            df[orig_col].to_dict()
        )
        
        # Merge with original data
        result_df = result_df.merge(
            df.reset_index()[df.columns.tolist() + ['index']],
            left_on='job_id',
            right_on='index',
            how='left'
        ).drop(columns=['index'])

        # Fix column names after merge
        if 'job_id_y' in result_df.columns:
            result_df = result_df.rename(columns={'job_id_y': 'job_id'})
        if 'job_id_x' in result_df.columns:
            result_df = result_df.drop(columns=['job_id_x'])
        
        # Display results
        print(f"\n  Optimized sampling complete:")
        print(f"    Total jobs selected: {len(result_df):,}")
        print(f"    Unique jobs: {len(selected_jobs):,}")
        print(f"    Domains filled: {sum(1 for c in domain_counts.values() if c > 0)}/{len(thresholds)}")
        
        # Show fulfillment
        print(f"\n  Domain fulfillment (top 10):")
        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        for domain, count in sorted_domains[:10]:
            target = thresholds[domain]
            pct = 100 * count / target if target > 0 else 0
            print(f"    {domain:20s}: {count:4d}/{target:4d} ({pct:.0f}%)")
        
        return result_df
    else:
        return pd.DataFrame()

def balance_jobs_with_domains(
    df: pd.DataFrame,
    thresholds: Dict[str, int],
    use_filtered_domains: bool = True,
    use_optimized: bool = True  # New parameter
) -> pd.DataFrame:
    """
    Balance jobs dataset using provided domain thresholds.
    
    Args:
        df: Jobs DataFrame
        thresholds: Dictionary mapping domains to target counts
        use_filtered_domains: Whether to use filtered domains for balancing
        use_optimized: Whether to use the optimized sampler (faster but less perfect)
        
    Returns:
        Balanced DataFrame
    """
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    print("\n" + "="*60)
    print("BALANCING JOBS DATASET")
    print("="*60)
    
    # Determine which domain column to use
    if use_filtered_domains and 'job_domain_filtered' in df.columns:
        domain_col = 'job_domain_filtered'
    elif use_filtered_domains and 'domains_filtered' in df.columns:
        domain_col = 'domains_filtered'
    else:
        domain_col = 'job_domain' if 'job_domain' in df.columns else 'domains'
    
    print(f"Using column: {domain_col}")
    print(f"Domains to balance: {len(thresholds)}")
    print(f"Target total jobs: {sum(thresholds.values()):,}")
    
    if use_optimized:
        result_df = optimized_job_sampler(df, domain_col, thresholds)
    else:
        # Use original randomized sampler
        result_df = randomized_assignment_sampler(
            df=df,
            domain_col=domain_col,
            thresholds=thresholds,
            subset_size=min(5000, len(df))
        )
    
    if len(result_df) > 0:
        print(f"\nFinal balanced dataset: {len(result_df):,} jobs")
    else:
        print("\nWarning: No jobs could be balanced")
    
    print("="*60 + "\n")
    
    return result_df

def randomized_assignment_sampler(
    df: pd.DataFrame,
    domain_col: str,
    thresholds: Dict[str, int],
    subset_size: int = 1000
) -> pd.DataFrame:
    """
    Assigns jobs using a randomized greedy approach for better performance.
    """
    available_jobs_df = df.copy()
    available_jobs_df['job_id'] = available_jobs_df.index
    
    assigned_jobs = []
    domain_counts = Counter()
    
    # Progress tracking
    total_needed = sum(thresholds.values())
    iterations = 0
    max_iterations = len(df) * 2  # Prevent infinite loops
    
    while iterations < max_iterations:
        iterations += 1
        
        domain_needs = Counter({
            domain: max(0, threshold - domain_counts[domain])
            for domain, threshold in thresholds.items()
        })
        
        if sum(domain_needs.values()) == 0:
            print("  ✓ All domain thresholds met")
            break

        if available_jobs_df.empty:
            print("  ⚠ Ran out of jobs before meeting all thresholds")
            break
            
        # Sample a subset of jobs for efficiency
        if len(available_jobs_df) > subset_size:
            job_pool = available_jobs_df.sample(n=subset_size)
        else:
            job_pool = available_jobs_df
            
        best_assignment = {'job_id': None, 'domain': None, 'score': -1}

        # Find best assignment in the pool
        for _, job in job_pool.iterrows():
            domains = job[domain_col]
            if not isinstance(domains, list):
                if isinstance(domains, str) and domains in thresholds:
                    domains = [domains]
                else:
                    continue
            
            if len(domains) == 0:
                continue
                
            job_flexibility = len(domains)
            for domain in domains:
                if domain in thresholds:
                    score = domain_needs[domain] + (1 / job_flexibility)
                    if score > best_assignment['score']:
                        best_assignment['score'] = score
                        best_assignment['job_id'] = job['job_id']
                        best_assignment['domain'] = domain
        
        if best_assignment['score'] <= 0:
            print("  ⚠ No useful assignments in current sample")
            break

        job_id_to_assign = best_assignment['job_id']
        domain_to_assign_to = best_assignment['domain']
        
        # Get original domains if available
        orig_domain_col = 'job_domain_original' if 'job_domain_original' in df.columns else 'domains_original' if 'domains_original' in df.columns else domain_col
        original_domains = df.loc[job_id_to_assign, orig_domain_col] if orig_domain_col in df.columns else df.loc[job_id_to_assign, domain_col]
        
        assigned_jobs.append({
            'job_id': job_id_to_assign,
            'assigned_domain': domain_to_assign_to,
            'original_domains': original_domains
        })
        
        domain_counts[domain_to_assign_to] += 1
        available_jobs_df = available_jobs_df[available_jobs_df['job_id'] != job_id_to_assign]
        
        # Progress update
        if len(assigned_jobs) % 1000 == 0:
            assigned_pct = 100 * len(assigned_jobs) / total_needed
            print(f"  Progress: {len(assigned_jobs):,}/{total_needed:,} ({assigned_pct:.1f}%)")

    # Create result DataFrame
    if assigned_jobs:
        result_df = pd.DataFrame(assigned_jobs)
        
        # Add all original columns from the source DataFrame
        result_df = result_df.merge(
            df.reset_index()[df.columns.tolist() + ['index']],
            left_on='job_id',
            right_on='index',
            how='left'
        ).drop(columns=['index'])

        # Fix column names after merge
        if 'job_id_y' in result_df.columns:
            result_df = result_df.rename(columns={'job_id_y': 'job_id'})
        if 'job_id_x' in result_df.columns:
            result_df = result_df.drop(columns=['job_id_x'])
        
        # Display domain distribution
        print(f"\n  Domain distribution in balanced dataset:")
        domain_dist = Counter(result_df['assigned_domain'])
        for domain, count in domain_dist.most_common(10):
            target = thresholds.get(domain, 0)
            print(f"    {domain:20s}: {count:,}/{target:,} jobs")
        
        return result_df
    else:
        print("  ⚠ No jobs could be assigned")
        return pd.DataFrame()