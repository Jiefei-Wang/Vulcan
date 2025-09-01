#!/usr/bin/env python3
"""
Verification script for Condition Matching Test.xlsx

This script verifies three main aspects:
1. Label correctness
2. Query and corpus name-id alignment  
3. Meaningless matches detection (identical names)

Usage:
    python verify_condition_matching.py

Requirements:
    pip install openpyxl pandas
"""

import pandas as pd


class ConditionMatchingVerifier:
    def __init__(self, excel_file: str = "verify_dataset/Condition Matching Train Subset.xlsx", 
                 matching_map_file: str = "data/matching/matching_map_table.feather",
                 name_table_file: str = "data/matching/condition_matching_name_table_train.feather"):
        """Initialize verifier with Excel file path and matching map table"""
        self.excel_file = excel_file
        self.matching_map_file = matching_map_file
        self.name_table_file = name_table_file
        self.df = None
        self.concept_table = None
        self.concept_synonym = None
        self.name_table = None
        self.name_id_to_name = {}
        self.name_id_to_source = {}
        self.name_id_to_source_id = {}
        self.matching_map_table = None
        self.matching_pairs = None
        self.source_id_to_concept_id = {}
        self.issues = {
            'label_issues': [],
            'name_id_misalignment': [],
            'identical_matches': [],
            'non_omop_rows': [],
            'non_omop_label_rows': []
        }
        
    def load_data(self):
        """Load Excel file and reference concept table"""
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
        
        # Load custom name table for additional ID mappings
        try:
            self.name_table = pd.read_feather(self.name_table_file)
            print(f"Loaded custom name table: {self.name_table.shape[0]} entries")
            
            # Create lookups based on actual structure (name_id, name, source, source_id)
            if 'name_id' in self.name_table.columns and 'name' in self.name_table.columns:
                # Create name_id -> name mapping
                self.name_id_to_name = self.name_table.set_index('name_id')['name'].to_dict()
                
                # Create name_id -> source mapping 
                self.name_id_to_source = self.name_table.set_index('name_id')['source'].to_dict()
                
                # Create name_id -> source_id mapping
                if 'source_id' in self.name_table.columns:
                    self.name_id_to_source_id = self.name_table.set_index('name_id')['source_id'].to_dict()
                
                print(f"Created custom name table mappings for {len(self.name_id_to_name)} entries")
            else:
                print(f"Warning: Expected 'name_id' and 'name' columns in {self.name_table_file}")
                print(f"Available columns: {list(self.name_table.columns)}")
                
        except Exception as e:
            print(f"Warning: Could not load custom name table {self.name_table_file}: {e}")
            print("Continuing without custom name mappings...")
        
        # Load matching map table for label verification
        try:
            self.matching_map_table = pd.read_feather(self.matching_map_file)
            print(f"Loaded matching map table: {self.matching_map_table.shape[0]} rows")
            
            # Clean and prepare matching pairs
            self._prepare_matching_pairs()
            
        except Exception as e:
            print(f"Error loading matching map table: {e}")
            return False
            
        return True
    
    def _prepare_matching_pairs(self):
        """Prepare clean matching pairs from matching_map_table"""
        # Convert source_id to numeric and filter out non-numeric values
        self.matching_map_table['source_id_int'] = pd.to_numeric(
            self.matching_map_table['source_id'], errors='coerce'
        )
        
        # Keep only rows with valid numeric source_id
        matching_map_clean = self.matching_map_table[
            self.matching_map_table['source_id_int'].notna()
        ].copy()
        matching_map_clean['source_id_int'] = matching_map_clean['source_id_int'].astype('int64')
        
        # Create set of (source_id, concept_id) pairs for fast lookup
        self.matching_pairs = set(zip(
            matching_map_clean['source_id_int'], 
            matching_map_clean['concept_id']
        ))
        
        print(f"Created {len(self.matching_pairs)} valid matching pairs from {len(matching_map_clean)} clean rows")
    
    def verify_labels(self):
        """Verify label correctness using matching_map_table"""
        print("\nVerifying labels...")
        
        if 'label' not in self.df.columns:
            self.issues['label_issues'].append("Label column not found in Excel file")
            return
            
        # Check label values
        unique_labels = self.df['label'].unique()
        print(f"Unique label values: {unique_labels}")
        
        # Check for valid label values (0 = cannot match, 1 = can match)
        valid_labels = {0, 1}
        invalid_labels = set(unique_labels) - valid_labels
        
        if invalid_labels:
            self.issues['label_issues'].append(f"Invalid label values found: {invalid_labels}")
            
        # Check label distribution
        label_counts = self.df['label'].value_counts()
        print(f"Label distribution - 0 (cannot match): {label_counts.get(0, 0)}, 1 (can match): {label_counts.get(1, 0)}")
        
        # Check for missing labels
        null_labels = self.df['label'].isnull().sum()
        if null_labels > 0:
            self.issues['label_issues'].append(f"Found {null_labels} rows with missing labels")
        
        # Verify label correctness using matching_map_table
        if self.matching_pairs is not None:
            self._verify_label_accuracy()
    
    def _verify_label_accuracy(self):
        """Verify if labels match the matching_map_table relationships"""
        print("Verifying label accuracy against matching_map_table...")
        
        required_cols = ['query_id', 'corpus_id', 'label']
        if not all(col in self.df.columns for col in required_cols):
            self.issues['label_issues'].append(f"Required columns for label verification not found: {required_cols}")
            return
        
        correct_labels = 0
        incorrect_labels = 0
        
        for idx, row in self.df.iterrows():
            query_id = row['query_id']  # This is name_id from custom table
            corpus_id = row['corpus_id']
            label = row['label']
            
            # Get source_id from custom name table using query_id (name_id)
            source_id = self.name_id_to_source_id.get(query_id)
            source = self.name_id_to_source.get(query_id)
            
            # Skip label verification for non-OMOP sources - export for manual review
            if source is not None and source != 'OMOP':
                query_name = row.get('query_name', 'N/A')
                corpus_name = row.get('corpus_name', 'N/A')
                self.issues['non_omop_label_rows'].append({
                    'row': idx,
                    'query_id': query_id,
                    'query_name': query_name,
                    'source': source,
                    'source_id': source_id,
                    'corpus_id': corpus_id,
                    'corpus_name': corpus_name,
                    'label': label
                })
                continue
            
            if source_id is not None:
                # Convert source_id to int if it's a string
                try:
                    source_id_int = int(source_id)
                except (ValueError, TypeError):
                    source_id_int = source_id
                
                # Check if (source_id, corpus_id) pair exists in matching_pairs
                # This handles cases where source_id maps to multiple concept_ids
                pair_exists = (source_id_int, corpus_id) in self.matching_pairs
            else:
                # query_id not found in custom name table
                pair_exists = False
            
            # Label should be 1 if pair exists, 0 if not
            expected_label = 1 if pair_exists else 0
            
            if label == expected_label:
                correct_labels += 1
            else:
                incorrect_labels += 1
                # Find all concept_ids that this source_id maps to for debugging
                if 'source_id_int' in locals():
                    mapped_concept_ids = [concept_id for (src_id, concept_id) in self.matching_pairs if src_id == source_id_int]
                else:
                    mapped_concept_ids = []
                    
                self.issues['label_issues'].append({
                    'type': 'accuracy_error',
                    'row': idx,
                    'query_id': query_id,
                    'corpus_id': corpus_id,
                    'actual_label': label,
                    'expected_label': expected_label,
                    'source_id': source_id,
                    'source_id_int': source_id_int if 'source_id_int' in locals() else source_id,
                    'mapped_concept_ids': mapped_concept_ids,
                    'pair_exists': pair_exists
                })
        
        # Calculate accuracy (excluding non-OMOP rows)
        total_verified = correct_labels + incorrect_labels
        non_omop_count = len(self.issues['non_omop_label_rows'])
        
        accuracy = correct_labels / total_verified * 100 if total_verified > 0 else 0
        print(f"Label accuracy (OMOP only): {correct_labels}/{total_verified} ({accuracy:.2f}%)")
        print(f"Non-OMOP labels skipped: {non_omop_count} (exported to Excel for manual review)")
        
        if incorrect_labels > 0:
            self.issues['label_issues'].append(f"Found {incorrect_labels} incorrect OMOP labels out of {total_verified} verified")
            
    def verify_name_id_alignment(self):
        """Verify query and corpus name-id alignment using custom name table"""
        print("\nVerifying name-id alignment...")
        
        # Check if columns exist
        required_cols = ['query_id', 'query_name', 'corpus_id', 'corpus_name']
        if not all(col in self.df.columns for col in required_cols):
            self.issues['name_id_misalignment'].append(
                f"Required columns not found. Available columns: {list(self.df.columns)}"
            )
            return
            
        print(f"Using columns: {required_cols}")
        
        # Verify query name-id alignment using custom name table
        query_misaligned = 0
        corpus_misaligned = 0
        non_omop_count = 0
        
        for idx, row in self.df.iterrows():
            query_id = row['query_id']
            query_name = row['query_name']
            corpus_id = row['corpus_id']
            corpus_name = row['corpus_name']
            
            # Check query alignment using custom name table
            if pd.notna(query_id):
                # Look up query_id in custom name table
                if query_id in self.name_id_to_source:
                    source = self.name_id_to_source[query_id]
                    expected_name = self.name_id_to_name.get(query_id, 'Not found')
                    
                    if source != 'OMOP':
                        # Extract non-OMOP rows for manual review
                        non_omop_count += 1
                        self.issues['non_omop_rows'].append({
                            'row': idx,
                            'query_id': query_id,
                            'query_name': query_name,
                            'source': source,
                            'expected_name': expected_name,
                            'corpus_id': corpus_id,
                            'corpus_name': corpus_name
                        })
                    else:
                        # For OMOP source, verify name alignment
                        # The names should match exactly since this is from the custom name table
                        # Handle potential string truncation/whitespace issues
                        if str(expected_name).strip() != str(query_name).strip():
                            query_misaligned += 1
                            # Try to find correct source_id for the provided name
                            source_id = self.name_id_to_source_id.get(query_id, 'Not found')
                            self.issues['name_id_misalignment'].append({
                                'row': idx,
                                'type': 'query',
                                'name_id': query_id,
                                'provided_name': query_name,
                                'expected_name': expected_name,
                                'source': source,
                                'source_id': source_id,
                                'expected_length': len(str(expected_name)),
                                'provided_length': len(str(query_name))
                            })
                else:
                    # Query ID not found in custom name table
                    query_misaligned += 1
                    self.issues['name_id_misalignment'].append({
                        'row': idx,
                        'type': 'query', 
                        'name_id': query_id,
                        'provided_name': query_name,
                        'expected_name': 'ID not found in custom name table',
                        'source': 'Unknown',
                        'source_id': 'Unknown'
                    })
                    
            # Check corpus alignment (standard OMOP concept table)
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
                    
        print(f"Query name-id misalignments (OMOP only): {query_misaligned}")
        print(f"Corpus name-id misalignments: {corpus_misaligned}")
        print(f"Non-OMOP query rows for manual review: {non_omop_count}")
        
    def detect_meaningless_matches(self):
        """Detect meaningless matches (identical or very similar names)"""
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
            query_name = str(row[query_col]).lower().strip()
            corpus_name = str(row[corpus_col]).lower().strip()
            
            # Check for identical matches only
            if query_name == corpus_name:
                identical_matches += 1
                self.issues['identical_matches'].append({
                    'row': idx,
                    'query_name': row[query_col],
                    'corpus_name': row[corpus_col]
                })
                    
        print(f"Identical matches: {identical_matches}")
        
    def generate_report(self):
        """Generate comprehensive verification report"""
        print("\n" + "="*60)
        print("VERIFICATION REPORT")
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
                if isinstance(issue, dict) and issue.get('type') == 'accuracy_error':
                    print(f"   • Row {issue['row']}: query_id={issue['query_id']}, corpus_id={issue['corpus_id']}, "
                          f"actual={issue['actual_label']}, expected={issue['expected_label']}")
                    if 'source_id' in issue and 'mapped_concept_ids' in issue:
                        print(f"     Chain: name_id({issue['query_id']}) → source_id({issue['source_id']}) → concept_ids({issue['mapped_concept_ids']})")
                        print(f"     Pair exists: {issue.get('pair_exists', 'Unknown')}")
                else:
                    print(f"   • {issue}")
            print()
            
        # Non-OMOP rows summary (details exported to Excel)
        if self.issues['non_omop_rows']:
            print(f"NON-OMOP ROWS: {len(self.issues['non_omop_rows'])} rows exported to Excel for manual review")
            print()

        # Name-ID alignment issues (OMOP only)
        if self.issues['name_id_misalignment']:
            print("NAME-ID ALIGNMENT ISSUES (OMOP only):")
            
            # Group by unique issues to deduplicate
            unique_issues = {}
            for issue in self.issues['name_id_misalignment']:
                if isinstance(issue, dict):
                    if issue['type'] == 'query':
                        key = (issue.get('name_id', 'N/A'), issue['provided_name'], issue['expected_name'], issue['type'])
                    else:
                        key = (issue.get('concept_id', 'N/A'), issue['provided_name'], issue['expected_name'], issue['type'])
                    
                    if key not in unique_issues:
                        unique_issues[key] = issue
                    
            # Show first 10 unique issues
            for issue in list(unique_issues.values())[:10]:
                if issue['type'] == 'query':
                    print(f"   • query name_id {issue.get('name_id', 'N/A')}")
                    print(f"     Provided: '{issue['provided_name']}'")
                    print(f"     Expected: '{issue['expected_name']}'")
                    print(f"     Source: {issue.get('source', 'N/A')}")
                else:
                    print(f"   • corpus concept_id {issue.get('concept_id', 'N/A')}")
                    print(f"     Provided: '{issue['provided_name']}'")
                    print(f"     Expected: '{issue['expected_name']}'")
                    print(f"     Correct ID for provided: {issue.get('correct_id_for_provided', 'N/A')}")
                    
            if len(unique_issues) > 10:
                print(f"   ... and {len(unique_issues) - 10} more unique misalignment types")
            
            total_misalignments = len(self.issues['name_id_misalignment'])
            unique_misalignments = len(unique_issues)
            print(f"   Total: {total_misalignments} misalignments ({unique_misalignments} unique types)")
            print()
            
        # Identical matches
        if self.issues['identical_matches']:
            print("IDENTICAL MATCHES (Same Names):")
            for issue in self.issues['identical_matches'][:10]:  # Show first 10
                if isinstance(issue, dict):
                    print(f"   • Row {issue['row']}: '{issue['query_name']}'")
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
        report_file = f"{excel_basename}_verification_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("CONDITION MATCHING VERIFICATION REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Excel File: {self.excel_file}\n")
            f.write(f"Total Rows: {len(self.df)}\n")
            f.write(f"Timestamp: {pd.Timestamp.now()}\n\n")
            
            # Summary with breakdown
            total_issues = sum(len(issues) for issues in self.issues.values())
            f.write(f"SUMMARY: {total_issues} total issues found\n")
            f.write("Issue breakdown:\n")
            f.write(f"  - Label issues: {len(self.issues['label_issues'])}\n")
            f.write(f"  - Name-ID misalignments: {len(self.issues['name_id_misalignment'])}\n")
            f.write(f"  - Identical matches: {len(self.issues['identical_matches'])}\n")
            f.write(f"  - Non-OMOP rows: {len(self.issues['non_omop_rows'])}\n")
            f.write(f"  - Non-OMOP label rows: {len(self.issues['non_omop_label_rows'])}\n\n")
            
            # Detailed issues
            accuracy_errors = [issue for issue in self.issues['label_issues'] if isinstance(issue, dict) and issue.get('type') == 'accuracy_error']
            other_issues = [issue for issue in self.issues['label_issues'] if not (isinstance(issue, dict) and issue.get('type') == 'accuracy_error')]
            
            # Write all accuracy errors
            if accuracy_errors:
                f.write(f"LABEL ACCURACY ERRORS ({len(accuracy_errors)} total):\n")
                f.write("-" * 40 + "\n")
                for issue in accuracy_errors:
                    f.write(f"Row {issue['row']}: query_id={issue['query_id']}, "
                           f"corpus_id={issue['corpus_id']}, actual={issue['actual_label']}, "
                           f"expected={issue['expected_label']}\n")
                    if 'source_id' in issue and 'mapped_concept_ids' in issue:
                        f.write(f"  Chain: name_id({issue['query_id']}) → source_id({issue['source_id']}) → concept_ids({issue['mapped_concept_ids']})\n")
                        f.write(f"  Pair exists: {issue.get('pair_exists', 'Unknown')}\n")
                    f.write("\n")
                f.write("\n")
            
            # Write other label issues
            for issue in other_issues:
                f.write(f"Label issue: {issue}\n")
            f.write("\n")
            
            if self.issues['non_omop_rows']:
                f.write("NON-OMOP ROWS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total non-OMOP rows: {len(self.issues['non_omop_rows'])}\n")
                f.write("Details exported to Excel file for manual review\n\n")
                
            if self.issues['non_omop_label_rows']:
                f.write("NON-OMOP LABEL ROWS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total non-OMOP label rows: {len(self.issues['non_omop_label_rows'])}\n")
                f.write("Details exported to Excel file for manual review\n\n")
                
            if self.issues['name_id_misalignment']:
                f.write("NAME-ID ALIGNMENT ISSUES (OMOP only):\n")
                f.write("-" * 30 + "\n")
                
                # Group by unique issues to deduplicate
                unique_issues = {}
                for issue in self.issues['name_id_misalignment']:
                    if isinstance(issue, dict):
                        if issue['type'] == 'query':
                            key = (issue.get('name_id', 'N/A'), issue['provided_name'], issue['expected_name'], issue['type'])
                        else:
                            key = (issue.get('concept_id', 'N/A'), issue['provided_name'], issue['expected_name'], issue['type'])
                        
                        if key not in unique_issues:
                            unique_issues[key] = issue
                
                # Write unique issues only
                for issue in unique_issues.values():
                    if issue['type'] == 'query':
                        f.write(f"query name_id {issue.get('name_id', 'N/A')}\n")
                        f.write(f"  Provided: '{issue['provided_name']}'\n")
                        f.write(f"  Expected: '{issue['expected_name']}'\n")
                        f.write(f"  Source: {issue.get('source', 'N/A')}\n\n")
                    else:
                        f.write(f"corpus concept_id {issue.get('concept_id', 'N/A')}\n")
                        f.write(f"  Provided: '{issue['provided_name']}'\n")
                        f.write(f"  Expected: '{issue['expected_name']}'\n")
                        f.write(f"  Correct ID for provided: {issue.get('correct_id_for_provided', 'N/A')}\n\n")
                
                # Write summary
                total_misalignments = len(self.issues['name_id_misalignment'])
                unique_misalignments = len(unique_issues)
                f.write(f"Total: {total_misalignments} misalignments ({unique_misalignments} unique types)\n\n")
                        
            if self.issues['identical_matches']:
                f.write("IDENTICAL MATCHES (Same Names):\n")
                f.write("-" * 30 + "\n")
                for issue in self.issues['identical_matches']:
                    if isinstance(issue, dict):
                        f.write(f"Row {issue['row']}: '{issue['query_name']}'\n")
                    else:
                        f.write(f"{issue}\n")
                f.write("\n")
                    
        print(f"Detailed report saved to: {report_file}")
    
    def export_non_omop_to_excel(self):
        """Export non-OMOP rows to Excel for manual review"""
        if not self.issues['non_omop_rows']:
            print("No non-OMOP rows found to export")
            return
            
        # Create unique Excel file name based on Excel file
        import os
        excel_basename = os.path.splitext(os.path.basename(self.excel_file))[0]
        excel_basename = excel_basename.replace(" ", "_").lower()
        export_file = f"{excel_basename}_non_omop_rows.xlsx"
        
        # Convert non-OMOP issues to DataFrame
        non_omop_data = []
        for issue in self.issues['non_omop_rows']:
            non_omop_data.append({
                'Row_Number': issue['row'],
                'Query_ID': issue['query_id'],
                'Query_Name_Provided': issue['query_name'],
                'Query_Name_Expected': issue['expected_name'],
                'Source': issue['source'],
                'Corpus_ID': issue['corpus_id'],
                'Corpus_Name': issue['corpus_name'],
                'Notes': f"Source is '{issue['source']}', not OMOP - requires manual review"
            })
        
        # Create DataFrame and export to Excel
        df_export = pd.DataFrame(non_omop_data)
        
        try:
            df_export.to_excel(export_file, index=False, engine='openpyxl')
            print(f"Non-OMOP rows exported to: {export_file}")
            print(f"Total non-OMOP rows: {len(non_omop_data)}")
            
            # Show breakdown by source
            source_counts = df_export['Source'].value_counts()
            print("Breakdown by source:")
            for source, count in source_counts.items():
                print(f"  {source}: {count} rows")
                
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            print("Make sure openpyxl is installed: pip install openpyxl")
    
    def export_non_omop_labels_to_excel(self):
        """Export non-OMOP label rows to Excel for manual review"""
        if not self.issues['non_omop_label_rows']:
            print("No non-OMOP label rows found to export")
            return
            
        # Create unique Excel file name based on Excel file
        import os
        excel_basename = os.path.splitext(os.path.basename(self.excel_file))[0]
        excel_basename = excel_basename.replace(" ", "_").lower()
        export_file = f"{excel_basename}_non_omop_labels.xlsx"
        
        # Convert non-OMOP label issues to DataFrame
        label_data = []
        for issue in self.issues['non_omop_label_rows']:
            label_data.append({
                'Row_Number': issue['row'],
                'Query_ID': issue['query_id'],
                'Query_Name': issue['query_name'],
                'Source': issue['source'],
                'Source_ID': issue['source_id'],
                'Corpus_ID': issue['corpus_id'],
                'Corpus_Name': issue['corpus_name'],
                'Label': issue['label'],
                'Notes': f"Source is '{issue['source']}', not OMOP - label verification skipped, requires manual review"
            })
        
        # Create DataFrame and export to Excel
        df_export = pd.DataFrame(label_data)
        
        try:
            df_export.to_excel(export_file, index=False, engine='openpyxl')
            print(f"Non-OMOP label rows exported to: {export_file}")
            print(f"Total non-OMOP label rows: {len(label_data)}")
            
            # Show breakdown by source
            source_counts = df_export['Source'].value_counts()
            print("Label breakdown by source:")
            for source, count in source_counts.items():
                print(f"  {source}: {count} rows")
                
        except Exception as e:
            print(f"Error exporting labels to Excel: {e}")
            print("Make sure openpyxl is installed: pip install openpyxl")
    
    def export_label_accuracy_errors_to_excel(self):
        """Export all label accuracy errors to Excel for detailed analysis"""
        # Get all accuracy errors
        accuracy_errors = [issue for issue in self.issues['label_issues'] 
                          if isinstance(issue, dict) and issue.get('type') == 'accuracy_error']
        
        if not accuracy_errors:
            print("No label accuracy errors found to export")
            return
            
        # Create unique Excel file name based on Excel file
        import os
        excel_basename = os.path.splitext(os.path.basename(self.excel_file))[0]
        excel_basename = excel_basename.replace(" ", "_").lower()
        export_file = f"{excel_basename}_label_accuracy_errors.xlsx"
        
        # Convert accuracy errors to DataFrame with additional details
        error_data = []
        for issue in accuracy_errors:
            # Get the original row data from the DataFrame
            original_row = self.df.iloc[issue['row']]
            
            error_data.append({
                'Row_Number': issue['row'],
                'Query_ID': issue['query_id'],
                'Query_Name': original_row.get('query_name', 'N/A'),
                'Corpus_ID': issue['corpus_id'],
                'Corpus_Name': original_row.get('corpus_name', 'N/A'),
                'Actual_Label': issue['actual_label'],
                'Expected_Label': issue['expected_label'],
                'Source_ID': issue['source_id'],
                'Source_ID_Int': issue.get('source_id_int', issue['source_id']),
                'Mapped_Concept_IDs': str(issue.get('mapped_concept_ids', [])),
                'Pair_Exists': issue.get('pair_exists', 'Unknown'),
                'Verification_Chain': f"name_id({issue['query_id']}) → source_id({issue['source_id']}) → concept_ids({issue.get('mapped_concept_ids', [])})",
                'Issue_Type': 'Label should be 0 but is 1' if issue['actual_label'] == 1 else 'Label should be 1 but is 0',
                'Notes': 'Concept mapping exists but label indicates no match' if issue['actual_label'] == 1 else 'No concept mapping found but label indicates match'
            })
        
        # Create DataFrame and export to Excel
        df_export = pd.DataFrame(error_data)
        
        try:
            df_export.to_excel(export_file, index=False, engine='openpyxl')
            print(f"Label accuracy errors exported to: {export_file}")
            print(f"Total label accuracy errors: {len(error_data)}")
            
            # Show breakdown by error type
            error_type_counts = df_export['Issue_Type'].value_counts()
            print("Error breakdown by type:")
            for error_type, count in error_type_counts.items():
                print(f"  {error_type}: {count} errors")
                
            return export_file
                
        except Exception as e:
            print(f"Error exporting label accuracy errors to Excel: {e}")
            print("Make sure openpyxl is installed: pip install openpyxl")
            return None
        
    def run_verification(self):
        """Run complete verification process"""
        print("Starting Condition Matching Verification")
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
        
        # Export non-OMOP rows to Excel
        self.export_non_omop_to_excel()
        
        # Export non-OMOP label rows to Excel
        self.export_non_omop_labels_to_excel()
        
        # Export label accuracy errors to Excel
        self.export_label_accuracy_errors_to_excel()
        
        return True


def main():
    """Main function"""
    verifier = ConditionMatchingVerifier()
    verifier.run_verification()


if __name__ == "__main__":
    main()