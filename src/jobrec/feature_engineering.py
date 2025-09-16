# Third-party library imports
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

def add_domain_bridge(resume_df: pd.DataFrame, bridge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain_bridge feature mapping resume categories to job domains.
    
    The bridge_df contains top 5 job domains for each resume category,
    with one domain per row. We need to aggregate them into lists.
    """
    
    # Create a copy to avoid modifying original
    resume_df = resume_df.copy()
    
    # Aggregate domains for each resume category into lists
    category_to_domains = {}
    
    for category in bridge_df['resume_category'].unique():
        # Get all domains for this category, sorted by rank
        category_domains = bridge_df[bridge_df['resume_category'] == category].sort_values('rank')
        domains_list = category_domains['job_domain'].tolist()
        category_to_domains[category] = domains_list
    
    # Add domain_bridge column
    resume_df['domain_bridge'] = resume_df['category'].map(category_to_domains)
    
    # Handle any unmapped categories
    resume_df['domain_bridge'] = resume_df['domain_bridge'].apply(
        lambda x: x if x is not None else []
    )
    
    # Display mappings
    print("\n" + "="*60)
    print("DOMAIN BRIDGE MAPPINGS")
    print("="*60)
    for cat in sorted(resume_df['category'].unique()[:10]):
        domains = category_to_domains.get(cat, [])
        print(f"  {cat:25s} -> {domains}")
    
    # Statistics
    total_categories = len(resume_df['category'].unique())
    mapped_categories = sum(1 for cat in resume_df['category'].unique() 
                          if cat in category_to_domains)
    print(f"\nMapped {mapped_categories}/{total_categories} categories")
    print("="*60 + "\n")
    
    return resume_df

def prepare_text_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Prepare final text features for modeling."""
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # List of numerical features to drop (handle various naming conventions)
    cols_to_drop = [
        'avg_word_length', 'unique_word_count', 'lexical_diversity', 
        'signature_term_count', 'skill_count', 'readability_grade', 
        'text_length', 'num_signature_terms', 'num_skills'
    ]
    
    # Drop columns that exist
    cols_to_drop_existing = [col for col in cols_to_drop if col in df_clean.columns]
    if cols_to_drop_existing:
        df_clean = df_clean.drop(columns=cols_to_drop_existing)
        print(f"Dropped {len(cols_to_drop_existing)} numerical feature columns")
    
    # Ensure clean text is string type
    if text_col in df_clean.columns:
        df_clean[text_col] = df_clean[text_col].astype(str)
    
    return df_clean

def identify_relevant_domains(
    resume_df: pd.DataFrame, 
    jobs_df: pd.DataFrame,
    min_job_threshold: float = 0.001,  # Minimum 0.1% of jobs
    min_jobs_count: int = 50  # Absolute minimum number of jobs
) -> Dict:
    """
    Identify relevant domains based on resume requirements and job availability.
    
    Returns:
        Dictionary with domain classification and statistics
    """
    
    # Get all domains from resume domain bridges
    resume_domains = set()
    if 'domain_bridge' in resume_df.columns:
        for domains_list in resume_df['domain_bridge'].dropna():
            if isinstance(domains_list, list):
                resume_domains.update(domains_list)
    
    # Count domain occurrences in jobs
    domain_col = 'job_domain' if 'job_domain' in jobs_df.columns else 'domains'
    domain_counts = {}
    total_jobs = len(jobs_df)
    
    for domains_list in jobs_df[domain_col].dropna():
        if isinstance(domains_list, list):
            for domain in domains_list:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        elif isinstance(domains_list, str):
            # Handle single domain as string
            domain_counts[domains_list] = domain_counts.get(domains_list, 0) + 1
    
    # Classify domains
    relevant_domains = []
    rare_domains = []
    
    for domain, count in domain_counts.items():
        percentage = count / total_jobs
        
        # A domain is relevant if it meets the threshold OR is needed by resumes
        if (percentage >= min_job_threshold and count >= min_jobs_count) or domain in resume_domains:
            relevant_domains.append(domain)
        else:
            rare_domains.append(domain)
    
    # Sort by frequency
    relevant_domains.sort(key=lambda x: domain_counts.get(x, 0), reverse=True)
    rare_domains.sort(key=lambda x: domain_counts.get(x, 0), reverse=True)
    
    # Print summary
    print("\n" + "="*60)
    print("DOMAIN CLASSIFICATION")
    print("="*60)
    print(f"Total unique domains: {len(domain_counts)}")
    print(f"Relevant domains: {len(relevant_domains)}")
    print(f"Rare domains: {len(rare_domains)}")
    print(f"Resume-required domains: {len(resume_domains)}")
    
    print(f"\nTop 10 relevant domains:")
    for domain in relevant_domains[:10]:
        count = domain_counts.get(domain, 0)
        pct = 100 * count / total_jobs
        print(f"  {domain:20s}: {count:5d} jobs ({pct:5.2f}%)")
    
    if rare_domains:
        print(f"\nRare domains (will be filtered):")
        for domain in rare_domains[:10]:
            count = domain_counts.get(domain, 0)
            pct = 100 * count / total_jobs
            print(f"  {domain:20s}: {count:5d} jobs ({pct:5.2f}%)")
    
    print("="*60 + "\n")
    
    return {
        'relevant_domains': relevant_domains,
        'rare_domains': rare_domains,
        'resume_domains': list(resume_domains),
        'domain_counts': domain_counts,
        'total_jobs': total_jobs
    }

def apply_rare_domain_filter(
    jobs_df: pd.DataFrame,
    domain_info: Dict,
    keep_original: bool = True
) -> pd.DataFrame:
    """
    Filter out rare domains from jobs and remove jobs with no remaining domains.
    """
    
    # Create a copy to avoid modifying original
    jobs_filtered = jobs_df.copy()
    
    relevant_domains = set(domain_info['relevant_domains'])
    rare_domains = set(domain_info['rare_domains'])
    
    # Determine domain column name
    domain_col = 'job_domain' if 'job_domain' in jobs_filtered.columns else 'domains'
    
    if keep_original:
        # Store original domains
        jobs_filtered[f'{domain_col}_original'] = jobs_filtered[domain_col].copy()
    
    # Filter domains for each job
    def filter_domains(domains_list):
        if not isinstance(domains_list, list):
            if isinstance(domains_list, str) and domains_list in relevant_domains:
                return [domains_list]
            return []
        return [d for d in domains_list if d in relevant_domains]
    
    jobs_filtered[f'{domain_col}_filtered'] = jobs_filtered[domain_col].apply(filter_domains)
    
    # Count jobs that would be removed
    initial_count = len(jobs_filtered)
    empty_domains = jobs_filtered[f'{domain_col}_filtered'].apply(lambda x: len(x) == 0)
    removed_count = empty_domains.sum()
    
    # Remove jobs with no relevant domains
    jobs_filtered = jobs_filtered[~empty_domains].copy()
    
    # Statistics
    print("\n" + "="*60)
    print("RARE DOMAIN FILTERING")
    print("="*60)
    print(f"Jobs before filter: {initial_count:,}")
    print(f"Jobs with only rare domains (removed): {removed_count:,}")
    print(f"Jobs after filter: {len(jobs_filtered):,}")
    print(f"Retention rate: {100 * len(jobs_filtered) / initial_count:.1f}%")
    print("="*60 + "\n")
    
    return jobs_filtered

def calculate_dynamic_thresholds(
    jobs_df: pd.DataFrame,
    domain_info: Dict,
    target_total_jobs: int = None,
    min_per_domain: int = 100,
    balance_strategy: str = 'equal_with_redistribution',  # New parameter
    use_filtered_domains: bool = True
) -> Dict[str, int]:
    """
    Calculate dynamic thresholds for job balancing with improved strategies.
    
    Args:
        jobs_df: Filtered jobs DataFrame
        domain_info: Output from identify_relevant_domains
        target_total_jobs: Target total number of jobs (None = auto-calculate)
        min_per_domain: Minimum jobs per domain
        balance_strategy: 'proportional', 'equal', or 'equal_with_redistribution'
        use_filtered_domains: Whether to use filtered domain column
        
    Returns:
        Dictionary mapping domains to their target counts
    """
    
    # Determine which domain column to use
    if use_filtered_domains and 'job_domain_filtered' in jobs_df.columns:
        domain_col = 'job_domain_filtered'
    elif use_filtered_domains and 'domains_filtered' in jobs_df.columns:
        domain_col = 'domains_filtered'
    else:
        domain_col = 'job_domain' if 'job_domain' in jobs_df.columns else 'domains'
    
    # Count actual available jobs per domain
    domain_availability = {}
    for domains_list in jobs_df[domain_col].dropna():
        if isinstance(domains_list, list):
            for domain in domains_list:
                domain_availability[domain] = domain_availability.get(domain, 0) + 1
        elif isinstance(domains_list, str):
            domain_availability[domains_list] = domain_availability.get(domains_list, 0) + 1
    
    # Get relevant domains and resume-required domains
    relevant_domains = domain_info['relevant_domains']
    resume_domains = set(domain_info['resume_domains'])
    
    # Filter to domains that actually have jobs
    domains_with_jobs = [d for d in relevant_domains if domain_availability.get(d, 0) > 0]
    
    # Auto-calculate target if not provided
    if target_total_jobs is None:
        # Aim for balanced representation: equal share per domain
        ideal_per_domain = min(500, max(min_per_domain, len(jobs_df) // len(domains_with_jobs)))
        target_total_jobs = ideal_per_domain * len(domains_with_jobs)
    
    # Ensure target doesn't exceed available jobs
    total_available = len(jobs_df)
    target_total_jobs = min(target_total_jobs, total_available)
    
    print("\n" + "="*60)
    print("DYNAMIC THRESHOLD CALCULATION")
    print("="*60)
    print(f"Strategy: {balance_strategy}")
    print(f"Domains with jobs: {len(domains_with_jobs)}")
    print(f"Resume-required domains: {len(resume_domains & set(domains_with_jobs))}")
    print(f"Total available jobs: {total_available:,}")
    print(f"Target total jobs: {target_total_jobs:,}")
    
    thresholds = {}
    
    if balance_strategy == 'proportional':
        # Original proportional strategy
        total_domain_occurrences = sum(domain_availability.get(d, 0) for d in domains_with_jobs)
        for domain in domains_with_jobs:
            available = domain_availability[domain]
            proportion = available / total_domain_occurrences
            target = int(target_total_jobs * proportion)
            target = max(min_per_domain, min(target, available))
            thresholds[domain] = target
            
    elif balance_strategy == 'equal':
        # Simple equal distribution
        equal_share = target_total_jobs // len(domains_with_jobs)
        for domain in domains_with_jobs:
            available = domain_availability[domain]
            target = min(equal_share, available)
            target = max(min_per_domain, target)
            thresholds[domain] = target
            
    elif balance_strategy == 'equal_with_redistribution':
        # Smart equal distribution with redistribution
        
        # Start with equal shares
        initial_share = target_total_jobs // len(domains_with_jobs)
        
        # Separate domains by priority (resume-required get priority)
        priority_domains = [d for d in domains_with_jobs if d in resume_domains]
        other_domains = [d for d in domains_with_jobs if d not in resume_domains]
        
        # First pass: assign initial shares respecting availability
        remaining_budget = target_total_jobs
        
        # Priority domains first
        for domain in priority_domains:
            available = domain_availability[domain]
            target = min(initial_share, available)
            target = max(min_per_domain, target)
            thresholds[domain] = target
            remaining_budget -= target
        
        # Other domains
        for domain in other_domains:
            available = domain_availability[domain]
            target = min(initial_share, available)
            target = max(min_per_domain, target)
            thresholds[domain] = target
            remaining_budget -= target
        
        # Second pass: redistribute remaining budget to domains with excess capacity
        iterations = 0
        while remaining_budget > 0 and iterations < 10:
            iterations += 1
            domains_with_capacity = [
                d for d in domains_with_jobs 
                if domain_availability[d] > thresholds[d]
            ]
            
            if not domains_with_capacity:
                break
            
            # Distribute remaining budget equally among domains with capacity
            extra_per_domain = remaining_budget // len(domains_with_capacity)
            if extra_per_domain == 0:
                extra_per_domain = 1
            
            for domain in domains_with_capacity:
                available = domain_availability[domain]
                current = thresholds[domain]
                can_add = min(extra_per_domain, available - current, remaining_budget)
                thresholds[domain] += can_add
                remaining_budget -= can_add
                
                if remaining_budget <= 0:
                    break
    
    # Final adjustment to ensure we don't exceed target
    total_threshold = sum(thresholds.values())
    if total_threshold > target_total_jobs:
        # Scale down proportionally
        scale_factor = target_total_jobs / total_threshold
        thresholds = {d: int(t * scale_factor) for d, t in thresholds.items()}
    
    # Display results
    print(f"\nThreshold Summary:")
    print(f"  Total threshold sum: {sum(thresholds.values()):,}")
    print(f"  Average per domain: {sum(thresholds.values()) // len(thresholds):,}")
    print(f"  Min threshold: {min(thresholds.values()):,}")
    print(f"  Max threshold: {max(thresholds.values()):,}")
    
    print(f"\nTop 10 domain thresholds:")
    sorted_thresholds = sorted(thresholds.items(), key=lambda x: x[1], reverse=True)
    for domain, threshold in sorted_thresholds[:10]:
        available = domain_availability.get(domain, 0)
        is_resume = "✓" if domain in resume_domains else " "
        utilization = 100 * threshold / available if available > 0 else 0
        print(f"  [{is_resume}] {domain:20s}: {threshold:4d}/{available:4d} ({utilization:.0f}% util)")
    
    print("="*60 + "\n")
    
    return thresholds