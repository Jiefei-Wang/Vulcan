#!/usr/bin/env python3
"""
Verification script for Relation Matching Excel files

This script verifies relationship concepts with special requirements:
1. Query names have prefixes like "<|parent of|>" that need to be stripped
2. Label=1 means query is parent of corpus, Label=0 means not parent
3. Uses concept_ancestor.feather for relationship verification
4. Same 5-column structure as matching verification

Usage:
    python verify_relation_matching.py

Requirements:
    pip install openpyxl pandas
"""

import pandas as pd
import re



class RelationMatchingVerifier:
    def __init__(self, excel_file: str = "verify_dataset/Condition Relation Train Subset.xlsx"):
        """Initialize verifier with Excel file path"""
        self.excel_file = excel_file
        self.df = None
        self.concept_table = None
        self.concept_synonym = None
        self.concept_ancestor = None
        self.ancestor_pairs = None
        self.issues = {
            'label_issues': [],
            'name_id_misalignment': [],
            'identical_matches': []
        }
        
    def load_data(self):
        """Load Excel file and reference tables"""
        print("Loading data...")
        
        # Load Excel file
        try:
            self.df = pd.read_excel(self.excel_file)
            print(f"Loaded Excel file: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            print(f"Columns: {list(self.df.columns)}")
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return False
            
        # Load concept reference table
        try:
            self.concept_table = pd.read_feather('data/omop_feather/concept.feather')
            print(f"Loaded concept table: {self.concept_table.shape[0]} concepts")
            
            # Create concept lookup for faster access
            self.concept_lookup = self.concept_table.set_index('concept_id')['concept_name'].to_dict()
            
            # Create reverse lookup: name -> concept_id (from main concept table)
            self.name_to_id_lookup = self.concept_table.set_index('concept_name')['concept_id'].to_dict()
            
            # Load concept_synonym table to include synonyms
            self.concept_synonym = pd.read_feather('data/omop_feather/concept_synonym.feather')
            print(f"Loaded concept synonym table: {self.concept_synonym.shape[0]} synonyms")
            
            # Add synonyms to the name_to_id_lookup
            synonym_lookup = self.concept_synonym.set_index('concept_synonym_name')['concept_id'].to_dict()
            self.name_to_id_lookup.update(synonym_lookup)
            
            print(f"Created concept lookup with {len(self.concept_lookup)} entries")
            print(f"Created name-to-id lookup with {len(self.name_to_id_lookup)} entries (includes synonyms)")
            
        except Exception as e:
            print(f"Error loading concept/synonym tables: {e}")
            return False
        
        # Load concept_ancestor table for relationship verification
        try:
            self.concept_ancestor = pd.read_feather('data/omop_feather/concept_ancestor.feather')
            print(f"Loaded concept ancestor table: {self.concept_ancestor.shape[0]} relationships")
            
            # Create set of (ancestor_id, descendant_id) pairs for fast lookup
            self.ancestor_pairs = set(zip(
                self.concept_ancestor['ancestor_concept_id'], 
                self.concept_ancestor['descendant_concept_id']
            ))
            print(f"Created {len(self.ancestor_pairs)} ancestor-descendant pairs")
            
        except Exception as e:
            print(f"Error loading concept ancestor table: {e}")
            return False
            
        return True
    
    def strip_relation_prefix(self, text):
        """Strip relation prefixes like '<|parent of|>' from text"""
        if pd.isna(text):
            return text
        
        # Remove prefixes like <|parent of|>, <|child of|>, etc.
        pattern = r'^<\|[^|]+\|>'
        cleaned_text = re.sub(pattern, '', str(text)).strip()
        return cleaned_text
    
    def verify_labels(self):
        """Verify label correctness using concept_ancestor relationships"""
        print("\nVerifying labels...")
        
        if 'label' not in self.df.columns:
            self.issues['label_issues'].append("Label column not found in Excel file")
            return
            
        # Check label values
        unique_labels = self.df['label'].unique()
        print(f"Unique label values: {unique_labels}")
        
        # Check for valid label values (0 = corpus not parent of query, 1 = corpus is parent of query)
        valid_labels = {0, 1}
        invalid_labels = set(unique_labels) - valid_labels
        
        if invalid_labels:
            self.issues['label_issues'].append(f"Invalid label values found: {invalid_labels}")
            
        # Check label distribution
        label_counts = self.df['label'].value_counts()
        print(f"Label distribution - 0 (corpus not parent of query): {label_counts.get(0, 0)}, 1 (corpus is parent of query): {label_counts.get(1, 0)}")
        
        # Check for missing labels
        null_labels = self.df['label'].isnull().sum()
        if null_labels > 0:
            self.issues['label_issues'].append(f"Found {null_labels} rows with missing labels")
        
        # Verify label correctness using concept_ancestor relationships
        if self.ancestor_pairs is not None:
            self._verify_relationship_accuracy()
    
    def _verify_relationship_accuracy(self):
        """Verify if labels match the parent-child relationships"""
        print("Verifying relationship accuracy against concept_ancestor...")
        
        required_cols = ['query_id', 'corpus_id', 'label']
        if not all(col in self.df.columns for col in required_cols):
            self.issues['label_issues'].append(f"Required columns for label verification not found: {required_cols}")
            return
        
        correct_labels = 0
        incorrect_labels = 0
        
        for idx, row in self.df.iterrows():
            query_id = row['query_id']
            corpus_id = row['corpus_id']
            label = row['label']
            
            # Check if corpus_id is ancestor of query_id (corpus is parent of query)
            # Since query has "<|parent of|>" prefix but is actually child of corpus
            corpus_is_parent = (corpus_id, query_id) in self.ancestor_pairs
            
            # Label should be 1 if corpus is parent of query, 0 if not
            expected_label = 1 if corpus_is_parent else 0
            
            if label == expected_label:
                correct_labels += 1
            else:
                incorrect_labels += 1
                if len([issue for issue in self.issues['label_issues'] if isinstance(issue, dict) and issue.get('type') == 'relationship_error']) < 10:
                    self.issues['label_issues'].append({
                        'type': 'relationship_error',
                        'row': idx,
                        'query_id': query_id,
                        'corpus_id': corpus_id,
                        'actual_label': label,
                        'expected_label': expected_label,
                        'relationship': 'corpus_is_parent_of_query' if corpus_is_parent else 'corpus_not_parent_of_query'
                    })
        
        accuracy = correct_labels / (correct_labels + incorrect_labels) * 100 if (correct_labels + incorrect_labels) > 0 else 0
        print(f"Relationship accuracy: {correct_labels}/{correct_labels + incorrect_labels} ({accuracy:.2f}%)")
        
        if incorrect_labels > 0:
            self.issues['label_issues'].append(f"Found {incorrect_labels} incorrect relationship labels out of {correct_labels + incorrect_labels} total")
            
    def verify_name_id_alignment(self):
        """Verify query and corpus name-id alignment (with prefix stripping for query)"""
        print("\nVerifying name-id alignment...")
        
        # Check if columns exist (try different possible column names)
        possible_col_names = [
            ['query_concept_id', 'query_concept_name', 'corpus_concept_id', 'corpus_concept_name'],
            ['concept_id1', 'sentence1', 'concept_id2', 'sentence2'],
            ['query_id', 'query_name', 'target_id', 'target_name'],
            ['query_id', 'query_name', 'corpus_id', 'corpus_name']
        ]
        
        col_mapping = None
        for cols in possible_col_names:
            if all(col in self.df.columns for col in cols):
                col_mapping = {
                    'query_id': cols[0],
                    'query_name': cols[1], 
                    'corpus_id': cols[2],
                    'corpus_name': cols[3]
                }
                break
                
        if not col_mapping:
            self.issues['name_id_misalignment'].append(
                f"Required columns not found. Available columns: {list(self.df.columns)}"
            )
            return
            
        print(f"Using column mapping: {col_mapping}")
        
        # Verify query name-id alignment (with prefix stripping)
        query_misaligned = 0
        corpus_misaligned = 0
        
        for idx, row in self.df.iterrows():
            query_id = row[col_mapping['query_id']]
            query_name_raw = row[col_mapping['query_name']]
            corpus_id = row[col_mapping['corpus_id']]
            corpus_name = row[col_mapping['corpus_name']]
            
            # Strip prefix from query name for comparison
            query_name_clean = self.strip_relation_prefix(query_name_raw)
            
            # Check query alignment (using cleaned name)
            if pd.notna(query_id) and query_id in self.concept_lookup:
                expected_query_name = self.concept_lookup[query_id]
                if expected_query_name != query_name_clean:
                    query_misaligned += 1
                    # Find what concept_id the provided name should map to
                    correct_id_for_provided = self.name_to_id_lookup.get(query_name_clean, 'Not found')
                    self.issues['name_id_misalignment'].append({
                        'row': idx,
                        'type': 'query',
                        'concept_id': query_id,
                        'provided_name': query_name_clean,
                        'provided_name_raw': query_name_raw,
                        'expected_name': expected_query_name,
                        'correct_id_for_provided': correct_id_for_provided
                    })
                    
            # Check corpus alignment (no prefix stripping needed)
            if pd.notna(corpus_id) and corpus_id in self.concept_lookup:
                expected_corpus_name = self.concept_lookup[corpus_id]
                if expected_corpus_name != corpus_name:
                    corpus_misaligned += 1
                    # Find what concept_id the provided name should map to
                    correct_id_for_provided = self.name_to_id_lookup.get(corpus_name, 'Not found')
                    self.issues['name_id_misalignment'].append({
                        'row': idx,
                        'type': 'corpus',
                        'concept_id': corpus_id,
                        'provided_name': corpus_name,
                        'expected_name': expected_corpus_name,
                        'correct_id_for_provided': correct_id_for_provided
                    })
                    
        print(f"Query name-id misalignments: {query_misaligned}")
        print(f"Corpus name-id misalignments: {corpus_misaligned}")
        
    def detect_meaningless_matches(self):
        """Detect meaningless matches (identical names after prefix stripping)"""
        print("\nDetecting meaningless matches...")
        
        # Try to find name columns
        name_cols = []
        for col in self.df.columns:
            if 'name' in col.lower() or 'sentence' in col.lower():
                name_cols.append(col)
                
        if len(name_cols) < 2:
            self.issues['identical_matches'].append("Could not find two name columns for comparison")
            return
            
        query_col = name_cols[0]
        corpus_col = name_cols[1] if len(name_cols) > 1 else name_cols[0]
        
        print(f"Comparing columns: '{query_col}' vs '{corpus_col}'")
        
        identical_matches = 0
        
        for idx, row in self.df.iterrows():
            query_name_raw = str(row[query_col])
            corpus_name = str(row[corpus_col]).lower().strip()
            
            # Strip prefix from query name and compare
            query_name_clean = self.strip_relation_prefix(query_name_raw).lower().strip()
            
            # Check for identical matches after prefix removal
            if query_name_clean == corpus_name:
                identical_matches += 1
                self.issues['identical_matches'].append({
                    'row': idx,
                    'query_name_raw': query_name_raw,
                    'query_name_clean': query_name_clean,
                    'corpus_name': row[corpus_col]
                })
                    
        print(f"Identical matches (after prefix removal): {identical_matches}")
        
    def generate_report(self):
        """Generate comprehensive verification report"""
        print("\n" + "="*60)
        print("RELATION MATCHING VERIFICATION REPORT")
        print(f"File: {self.excel_file}")
        print("="*60)
        
        total_issues = sum(len(issues) for issues in self.issues.values())
        
        if total_issues == 0:
            print("ALL VERIFICATIONS PASSED! No issues found.")
            return
            
        print(f"Found {total_issues} total issues:")
        print()
        
        # Label issues
        if self.issues['label_issues']:
            print("LABEL ISSUES:")
            for issue in self.issues['label_issues']:
                if isinstance(issue, dict) and issue.get('type') == 'relationship_error':
                    print(f"   • Row {issue['row']}: query_id={issue['query_id']}, corpus_id={issue['corpus_id']}, "
                          f"actual={issue['actual_label']}, expected={issue['expected_label']} ({issue['relationship']})")
                else:
                    print(f"   • {issue}")
            print()
            
        # Name-ID alignment issues
        if self.issues['name_id_misalignment']:
            print("NAME-ID ALIGNMENT ISSUES:")
            
            # Group by (concept_id, provided_name, expected_name) to deduplicate
            unique_issues = {}
            for issue in self.issues['name_id_misalignment']:
                if isinstance(issue, dict):
                    key = (issue['concept_id'], issue['provided_name'], issue['expected_name'], issue['type'])
                    if key not in unique_issues:
                        unique_issues[key] = issue
                    
            # Show first 10 unique issues
            for issue in list(unique_issues.values())[:10]:
                print(f"   • {issue['type']} concept_id {issue['concept_id']}")
                print(f"     Provided: '{issue['provided_name']}'")
                if issue.get('provided_name_raw'):
                    print(f"     Raw (with prefix): '{issue['provided_name_raw']}'")
                print(f"     Expected: '{issue['expected_name']}'")
                print(f"     Provided name should be ID: {issue['correct_id_for_provided']}")
                    
            if len(unique_issues) > 10:
                print(f"   ... and {len(unique_issues) - 10} more unique misalignment types")
            
            total_misalignments = len(self.issues['name_id_misalignment'])
            unique_misalignments = len(unique_issues)
            print(f"   Total: {total_misalignments} misalignments ({unique_misalignments} unique types)")
            print()
            
        # Identical matches
        if self.issues['identical_matches']:
            print("IDENTICAL MATCHES (Same Names after prefix removal):")
            for issue in self.issues['identical_matches'][:10]:  # Show first 10
                if isinstance(issue, dict):
                    print(f"   • Row {issue['row']}: '{issue['query_name_clean']}' (raw: '{issue['query_name_raw']}')")
                else:
                    print(f"   • {issue}")
                
            if len(self.issues['identical_matches']) > 10:
                print(f"   ... and {len(self.issues['identical_matches']) - 10} more")
            print()
            
        print("="*60)
        
        # Save detailed report to file
        self.save_detailed_report()
        
    def save_detailed_report(self):
        """Save detailed report to file"""
        # Create unique report file name based on Excel file
        import os
        excel_basename = os.path.splitext(os.path.basename(self.excel_file))[0]
        excel_basename = excel_basename.replace(" ", "_").lower()
        report_file = f"{excel_basename}_relation_verification_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("RELATION MATCHING VERIFICATION REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Excel File: {self.excel_file}\n")
            f.write(f"Total Rows: {len(self.df)}\n")
            f.write(f"Timestamp: {pd.Timestamp.now()}\n\n")
            
            # Summary
            total_issues = sum(len(issues) for issues in self.issues.values())
            f.write(f"SUMMARY: {total_issues} total issues found\n\n")
            
            # Detailed issues
            for issue in self.issues['label_issues']:
                if isinstance(issue, dict) and issue.get('type') == 'relationship_error':
                    f.write(f"Relationship error - Row {issue['row']}: query_id={issue['query_id']}, "
                           f"corpus_id={issue['corpus_id']}, actual={issue['actual_label']}, "
                           f"expected={issue['expected_label']} ({issue['relationship']})\n")
                else:
                    f.write(f"Label issue: {issue}\n")
            f.write("\n")
            
            if self.issues['name_id_misalignment']:
                f.write("NAME-ID ALIGNMENT ISSUES:\n")
                f.write("-" * 30 + "\n")
                
                # Group by (concept_id, provided_name, expected_name) to deduplicate
                unique_issues = {}
                for issue in self.issues['name_id_misalignment']:
                    if isinstance(issue, dict):
                        key = (issue['concept_id'], issue['provided_name'], issue['expected_name'], issue['type'])
                        if key not in unique_issues:
                            unique_issues[key] = issue
                
                # Write unique issues only
                for issue in unique_issues.values():
                    f.write(f"{issue['type']} concept_id {issue['concept_id']}\n")
                    f.write(f"  Provided: '{issue['provided_name']}'\n")
                    if issue.get('provided_name_raw'):
                        f.write(f"  Raw (with prefix): '{issue['provided_name_raw']}'\n")
                    f.write(f"  Expected: '{issue['expected_name']}'\n")
                    f.write(f"  Provided name should be ID: {issue['correct_id_for_provided']}\n\n")
                
                # Write summary
                total_misalignments = len(self.issues['name_id_misalignment'])
                unique_misalignments = len(unique_issues)
                f.write(f"Total: {total_misalignments} misalignments ({unique_misalignments} unique types)\n\n")
                        
            if self.issues['identical_matches']:
                f.write("IDENTICAL MATCHES (Same Names after prefix removal):\n")
                f.write("-" * 30 + "\n")
                for issue in self.issues['identical_matches']:
                    if isinstance(issue, dict):
                        f.write(f"Row {issue['row']}: '{issue['query_name_clean']}' (raw: '{issue['query_name_raw']}')\n")
                    else:
                        f.write(f"{issue}\n")
                f.write("\n")
                    
        print(f"Detailed report saved to: {report_file}")
        
    def run_verification(self):
        """Run complete verification process"""
        print("Starting Relation Matching Verification")
        print(f"File: {self.excel_file}")
        print("="*60)
        
        # Load data
        if not self.load_data():
            return False
            
        # Run all verifications
        self.verify_labels()
        self.verify_name_id_alignment() 
        self.detect_meaningless_matches()
        
        # Generate report
        self.generate_report()
        
        return True


def main():
    """Main function"""
    verifier = RelationMatchingVerifier()
    verifier.run_verification()


if __name__ == "__main__":
    main()