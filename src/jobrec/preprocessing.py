# Standard library imports
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Optional, List

# Third-party library imports
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FilterConfig:
    """Configuration for document filtering."""
    # Hard thresholds (applied universally)
    min_word_count: int = 50
    max_word_count: int = 5000
    min_skills: int = 1
    max_skills: int = 50  
    min_signature_terms: int = 1
    max_lexical_diversity: float = 0.8
    
    # Quantile thresholds (applied per class)
    unique_word_quantile: Tuple[float, float] = (0.025, 0.975)
    lexical_diversity_quantile: Tuple[float, float] = (0.025, 0.975)
    readability_quantile: Tuple[float, float] = (0.05, 0.95)
    
    # Minimum samples per class to apply quantile filtering
    min_class_samples: int = 20
    
    # Output configuration
    log_dir: str = './filter_logs'
    save_logs: bool = True

class DocumentFilter:
    """Advanced document filtration system with per-class quantile filtering."""
    
    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()
        self.filter_stats = {}
        self.summary_stats = {}
        
        # Setup logging directory
        if self.config.save_logs:
            self.log_dir = Path(self.config.log_dir)
            self.log_dir.mkdir(exist_ok=True)
    
    def apply_hard_filters(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Apply universal hard threshold filters."""
        initial_count = len(df)
        stage_counts = {'initial': initial_count}
        
        # Word count bounds
        df = df[
            (df['text_length'] >= self.config.min_word_count) &
            (df['text_length'] <= self.config.max_word_count)
        ]
        stage_counts['word_count'] = len(df)
        
        # Skills bounds
        skill_col = 'num_skills' if 'num_skills' in df.columns else 'skill_count'
        if skill_col in df.columns:
            df = df[
                (df[skill_col] >= self.config.min_skills) &
                (df[skill_col] <= self.config.max_skills)
            ]
            stage_counts['skills'] = len(df)
        
        # Signature terms minimum
        sig_col = 'num_signature_terms' if 'num_signature_terms' in df.columns else 'signature_term_count'
        if sig_col in df.columns:
            df = df[df[sig_col] >= self.config.min_signature_terms]
            stage_counts['signature_terms'] = len(df)
        
        # Lexical diversity maximum
        if 'lexical_diversity' in df.columns:
            df = df[df['lexical_diversity'] <= self.config.max_lexical_diversity]
            stage_counts['lexical_diversity'] = len(df)
        
        if verbose:
            print("  Hard filters applied:")
            prev_count = initial_count
            for stage, count in stage_counts.items():
                if stage != 'initial':
                    removed = prev_count - count
                    print(f"    After {stage:20s}: {count:6,} ({removed:,} removed)")
                    prev_count = count
        
        return df
    
    def apply_class_quantile_filters(self, df: pd.DataFrame, class_name: str) -> pd.DataFrame:
        """Apply quantile-based filters for a specific class."""
        if len(df) < self.config.min_class_samples:
            logger.info(f"Skipping quantile filters for {class_name} (only {len(df)} samples)")
            return df
        
        initial_count = len(df)
        
        # Unique word count quantiles
        lower_unique, upper_unique = df['unique_word_count'].quantile(
            self.config.unique_word_quantile
        )
        df = df[
            (df['unique_word_count'] >= lower_unique) &
            (df['unique_word_count'] <= upper_unique)
        ]
        
        # Lexical diversity quantiles
        lower_lex, upper_lex = df['lexical_diversity'].quantile(
            self.config.lexical_diversity_quantile
        )
        df = df[
            (df['lexical_diversity'] >= lower_lex) &
            (df['lexical_diversity'] <= upper_lex)
        ]
        
        # Readability grade quantiles
        lower_read, upper_read = df['readability_grade'].quantile(
            self.config.readability_quantile
        )
        df = df[
            (df['readability_grade'] >= lower_read) &
            (df['readability_grade'] <= upper_read)
        ]
        
        # Store statistics
        self.filter_stats[class_name] = {
            'unique_word_bounds': (lower_unique, upper_unique),
            'lexical_diversity_bounds': (lower_lex, upper_lex),
            'readability_bounds': (lower_read, upper_read),
            'samples_before': initial_count,
            'samples_after': len(df),
            'retention_rate': len(df) / initial_count if initial_count > 0 else 0
        }
        
        return df
    
    def filter_resumes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter resume dataset with per-category quantile filtering."""
        print("\n" + "="*60)
        print("FILTERING RESUMES")
        print("="*60)
        
        initial_count = len(df)
        
        # Apply hard filters
        print("\n1. Applying hard filters...")
        df_filtered = self.apply_hard_filters(df.copy())
        hard_filter_retained = len(df_filtered)
        
        # Apply quantile filters per category
        print("\n2. Applying per-category quantile filters...")
        filtered_dfs = []
        category_stats = []
        
        for category in sorted(df_filtered['category'].unique()):
            category_df = df_filtered[df_filtered['category'] == category]
            initial_cat_count = len(category_df)
            
            # Apply class-specific quantile filters
            category_filtered = self.apply_class_quantile_filters(category_df, category)
            filtered_dfs.append(category_filtered)
            
            # Collect stats
            retention = len(category_filtered) / initial_cat_count if initial_cat_count > 0 else 0
            category_stats.append({
                'category': category,
                'initial': initial_cat_count,
                'final': len(category_filtered),
                'retention': retention
            })
        
        # Combine all filtered categories
        final_df = pd.concat(filtered_dfs, ignore_index=True)
        
        # Display summary
        print("\n3. Category-level summary:")
        print(f"   {'Category':25s} {'Initial':>8s} {'Final':>8s} {'Retained':>10s}")
        print("   " + "-"*53)
        for stat in category_stats[:10]:  # Show first 10
            print(f"   {stat['category']:25s} {stat['initial']:8,d} {stat['final']:8,d} {stat['retention']:9.1%}")
        
        if len(category_stats) > 10:
            print(f"   ... and {len(category_stats) - 10} more categories")
        
        # Overall summary
        self.summary_stats['resumes'] = {
            'initial_count': initial_count,
            'after_hard_filters': hard_filter_retained,
            'final_count': len(final_df),
            'overall_retention': len(final_df) / initial_count if initial_count > 0 else 0,
            'categories_processed': len(category_stats)
        }
        
        print(f"\n4. Overall summary:")
        print(f"   Initial resumes:      {initial_count:8,}")
        print(f"   After hard filters:   {hard_filter_retained:8,} ({100*hard_filter_retained/initial_count:.1f}%)")
        print(f"   After quantile filters: {len(final_df):8,} ({100*len(final_df)/initial_count:.1f}%)")
        print("="*60 + "\n")
        
        # Save logs if configured
        if self.config.save_logs:
            self._save_filter_logs('resumes', category_stats)
        
        return final_df
    
    def filter_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter jobs dataset with per-domain quantile filtering."""
        print("\n" + "="*60)
        print("FILTERING JOBS")
        print("="*60)
        
        initial_count = len(df)
        
        # Determine domain column
        domain_col = 'job_domain' if 'job_domain' in df.columns else 'domains'
        
        # Apply hard filters
        print("\n1. Applying hard filters...")
        df_filtered = self.apply_hard_filters(df.copy())
        hard_filter_retained = len(df_filtered)
        
        # Store original domains
        df_filtered['original_index'] = df_filtered.index
        original_domains = df_filtered[['job_id', domain_col]].set_index('job_id')[domain_col].to_dict()
        
        # Explode domains
        print("\n2. Exploding domains for per-domain filtering...")
        df_exploded = df_filtered.explode(domain_col)
        unique_domains = df_exploded[domain_col].unique()
        print(f"   Found {len(unique_domains)} unique domains")
        print(f"   Exploded to {len(df_exploded):,} job-domain pairs")
        
        # Apply quantile filters per domain
        print("\n3. Applying per-domain quantile filters...")
        filtered_dfs = []
        domain_stats = []
        
        # Sort domains by frequency for better display
        domain_counts = df_exploded[domain_col].value_counts()
        
        for domain in domain_counts.index:
            domain_df = df_exploded[df_exploded[domain_col] == domain]
            initial_domain_count = len(domain_df)
            
            # Apply domain-specific quantile filters
            domain_filtered = self.apply_class_quantile_filters(domain_df, domain)
            filtered_dfs.append(domain_filtered)
            
            # Collect stats
            retention = len(domain_filtered) / initial_domain_count if initial_domain_count > 0 else 0
            domain_stats.append({
                'domain': domain,
                'initial': initial_domain_count,
                'final': len(domain_filtered),
                'retention': retention
            })
        
        # Combine and de-duplicate
        print("\n4. Combining and de-duplicating...")
        if filtered_dfs:
            combined_df = pd.concat(filtered_dfs, ignore_index=True)
            
            # Remove duplicates
            final_df = combined_df.drop_duplicates(subset=['job_id'], keep='first')
            
            # Restore original domain lists
            final_df[domain_col] = final_df['job_id'].map(original_domains)
            
            # Clean up
            if 'original_index' in final_df.columns:
                final_df = final_df.drop(columns=['original_index'])
            
            # Display domain summary
            print("\n5. Domain-level summary (top 20):")
            print(f"   {'Domain':20s} {'Initial':>8s} {'Final':>8s} {'Retained':>10s}")
            print("   " + "-"*48)
            for stat in domain_stats[:20]:
                print(f"   {stat['domain']:20s} {stat['initial']:8,d} {stat['final']:8,d} {stat['retention']:9.1%}")
            
            if len(domain_stats) > 20:
                print(f"   ... and {len(domain_stats) - 20} more domains")
            
            # Overall summary
            self.summary_stats['jobs'] = {
                'initial_count': initial_count,
                'after_hard_filters': hard_filter_retained,
                'final_count': len(final_df),
                'overall_retention': len(final_df) / initial_count if initial_count > 0 else 0,
                'domains_processed': len(domain_stats),
                'unique_jobs': len(final_df)
            }
            
            print(f"\n6. Overall summary:")
            print(f"   Initial jobs:         {initial_count:8,}")
            print(f"   After hard filters:   {hard_filter_retained:8,} ({100*hard_filter_retained/initial_count:.1f}%)")
            print(f"   After quantile filters: {len(final_df):8,} ({100*len(final_df)/initial_count:.1f}%)")
            print(f"   Unique jobs retained: {len(final_df):8,}")
            print("="*60 + "\n")
            
            # Save logs if configured
            if self.config.save_logs:
                self._save_filter_logs('jobs', domain_stats)
            
            return final_df.reset_index(drop=True)
        else:
            print("Warning: No jobs passed filtering!")
            return pd.DataFrame()
    
    def apply_filters(self, df: pd.DataFrame, doc_type: str = 'resume') -> pd.DataFrame:
        """Main entry point for filtering."""
        if doc_type == 'resume':
            return self.filter_resumes(df)
        elif doc_type == 'job':
            return self.filter_jobs(df)
        else:
            raise ValueError(f"Unknown doc_type: {doc_type}. Use 'resume' or 'job'")
    
    def get_filter_summary(self) -> pd.DataFrame:
        """Return summary of filtering statistics per class."""
        if not self.filter_stats:
            return pd.DataFrame()
        
        summary_data = []
        for class_name, stats in self.filter_stats.items():
            summary_data.append({
                'class': class_name,
                'samples_before': stats['samples_before'],
                'samples_after': stats['samples_after'],
                'retention_rate': stats['retention_rate'],
                'unique_word_lower': stats['unique_word_bounds'][0],
                'unique_word_upper': stats['unique_word_bounds'][1],
                'lexical_div_lower': stats['lexical_diversity_bounds'][0],
                'lexical_div_upper': stats['lexical_diversity_bounds'][1],
                'readability_lower': stats['readability_bounds'][0],
                'readability_upper': stats['readability_bounds'][1]
            })
        
        return pd.DataFrame(summary_data)
    
    def _save_filter_logs(self, doc_type: str, class_stats: List[Dict]):
        """Save filtering logs to CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed class statistics
        class_df = pd.DataFrame(class_stats)
        class_file = self.log_dir / f"filter_{doc_type}_classes_{timestamp}.csv"
        class_df.to_csv(class_file, index=False)
        
        # Save filter configuration
        config_data = {
            'timestamp': timestamp,
            'doc_type': doc_type,
            'min_word_count': self.config.min_word_count,
            'max_word_count': self.config.max_word_count,
            'min_skills': self.config.min_skills,
            'max_skills': self.config.max_skills,
            'min_signature_terms': self.config.min_signature_terms,
            'max_lexical_diversity': self.config.max_lexical_diversity,
            'unique_word_quantile_lower': self.config.unique_word_quantile[0],
            'unique_word_quantile_upper': self.config.unique_word_quantile[1],
            'lexical_diversity_quantile_lower': self.config.lexical_diversity_quantile[0],
            'lexical_diversity_quantile_upper': self.config.lexical_diversity_quantile[1],
            'readability_quantile_lower': self.config.readability_quantile[0],
            'readability_quantile_upper': self.config.readability_quantile[1],
            'min_class_samples': self.config.min_class_samples
        }
        
        # Add summary statistics
        if doc_type in self.summary_stats:
            config_data.update(self.summary_stats[doc_type])
        
        # Save configuration and summary
        config_file = self.log_dir / f"filter_{doc_type}_config_{timestamp}.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Save overall metrics (overwrite each time)
        metrics_file = self.log_dir / f"filter_{doc_type}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved filter logs to {self.log_dir}")