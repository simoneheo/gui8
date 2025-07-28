import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import re
import time
from collections import Counter
from file import File, FileStatus
from channel import Channel, SourceType
from parse_config import ParseConfig, ParseStrategy, ParseResult, DataType


class AutoParser:
    """
    Clean, robust auto-parser that detects metadata, headers, time columns, and categorical data.
    Creates File and Channel objects with proper time column detection.
    """
    
    def __init__(self):
        self.last_error = None
        self.parse_stats = {
            'files_parsed': 0,
            'parse_time_ms': 0,
            'metadata_lines_skipped': 0,
            'strategies_used': Counter()
        }
    
    def parse_file(self, file_path: Path) -> Tuple[Optional[File], List[Channel]]:
        """
        Main entry point - parse a file and return File object + Channel list.
        
        Args:
            file_path: Path to file to parse
            
        Returns:
            Tuple of (File object, List of Channel objects)
        """
        start_time = time.time()
        
        try:
            # Create File object with parse config
            file_obj = File(file_path)
            file_obj.state.set_status(FileStatus.NOT_PARSED)
            
            # Read file and detect structure
            lines, encoding = self._read_file_with_encoding(file_path)
            if not lines:
                file_obj.state.set_error("Could not read file or file is empty")
                return file_obj, []
            
            # Detect and skip metadata lines
            metadata_skip = self._detect_metadata_lines(lines)
            data_lines = lines[metadata_skip:]
            self.parse_stats['metadata_lines_skipped'] += metadata_skip
            
            if not data_lines:
                file_obj.state.set_error("No data lines found after metadata")
                return file_obj, []
            
            # Detect structure (delimiter, header, column types)
            structure_info = self._detect_structure(data_lines)
            if not structure_info:
                file_obj.state.set_error("Could not detect file structure")
                return file_obj, []
            
            # Parse data into DataFrame
            df = self._parse_data(data_lines, structure_info)
            if df is None or df.empty:
                file_obj.state.set_error("Failed to parse data into DataFrame")
                return file_obj, []
            
            # Detect X (time) column
            x_col_info = self._detect_x_column(df, structure_info.get('headers', []))
            
            # Create channels from DataFrame
            channels = self._create_channels(df, file_obj, x_col_info, structure_info)
            
            # Update file status and results
            if channels:
                file_obj.state.set_status(FileStatus.PARSED)
                
                # Store parse results
                parse_result = ParseResult(
                    success=True,
                    strategy_used=ParseStrategy.PANDAS_AUTO,  # Could be more specific
                    rows_parsed=len(df),
                    columns_found=len(df.columns),
                    channels_created=len(channels),
                    parse_time_ms=(time.time() - start_time) * 1000,
                    encoding_detected=encoding,
                    metadata_lines_skipped=metadata_skip
                )
                file_obj.parse_config.set_parse_result(parse_result)
                
                # Store column information
                for col in df.columns:
                    data_type = self._determine_data_type(df[col])
                    is_time = col == x_col_info.get('column_name') if x_col_info else False
                    file_obj.parse_config.add_column_info(
                        name=str(col),
                        data_type=data_type,
                        is_time_column=is_time,
                        row_count=len(df),
                        null_count=df[col].isnull().sum()
                    )
            else:
                file_obj.state.set_error("No valid channels created from data")
            
            # Update global stats
            elapsed_ms = (time.time() - start_time) * 1000
            self.parse_stats['files_parsed'] += 1
            self.parse_stats['parse_time_ms'] += elapsed_ms
            self.parse_stats['strategies_used'][structure_info.get('strategy', 'unknown')] += 1
            
            return file_obj, channels
            
        except Exception as e:
            self.last_error = str(e)
            if 'file_obj' not in locals():
                file_obj = File(file_path)
            file_obj.state.set_error(f"Parse error: {str(e)}")
            return file_obj, []
    
    def _read_file_with_encoding(self, file_path: Path) -> Tuple[List[str], str]:
        """
        Read file with automatic encoding detection.
        """
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    lines = [line.rstrip('\r\n') for line in f.readlines()]
                    return [line for line in lines if line or not line.isspace()], encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # Final fallback
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                decoded = content.decode('utf-8', errors='replace')
                lines = [line.strip() for line in decoded.split('\n') if line.strip()]
                return lines, 'utf-8-fallback'
        except Exception:
            return [], 'unknown'
    
    def _detect_metadata_lines(self, lines: List[str]) -> int:
        """
        Detect and count metadata lines at the beginning of file.
        Based on structure_detector.py approach.
        """
        metadata_prefixes = ('#', '//', '%', ';', '--', '"""', ';;;', '>', '::')
        section_pattern = re.compile(r'^(={3,}|-{3,}|\*{3,}|/{3,})\s*$')
        
        skip_count = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            # Check for metadata prefixes
            if any(stripped.startswith(prefix) for prefix in metadata_prefixes):
                skip_count += 1
                continue
            
            # Check for section dividers
            if section_pattern.match(stripped):
                skip_count += 1
                continue
            
            # Check for key-value patterns (Subject: value, Rate = 100 Hz)
            if self._is_metadata_key_value(stripped):
                skip_count += 1
                continue
            
            # Found non-metadata line, stop
            break
        
        return skip_count
    
    def _is_metadata_key_value(self, line: str) -> bool:
        """
        Check if line looks like metadata key-value pair.
        Enhanced for non-ASCII metadata.
        """
        # Patterns like "Subject: John", "Sampling Rate = 100 Hz", "Device :: AB123"
        # Updated patterns to handle non-ASCII characters
        kv_patterns = [
            r'^[A-Za-z\s\u00C0-\u024F\u1E00-\u1EFF\u4E00-\u9FFF\u0400-\u04FF]{1,30}[:=]{1,2}\s*.+$',  # Key: Value or Key = Value (with Unicode)
            r'^[A-Za-z\s\u00C0-\u024F\u1E00-\u1EFF\u4E00-\u9FFF\u0400-\u04FF]{1,30}::\s*.+$'          # Key :: Value (with Unicode)
        ]
        
        for pattern in kv_patterns:
            if re.match(pattern, line):
                # Additional check: key shouldn't be all numeric
                key_part = line.split(':', 1)[0].strip() if ':' in line else line.split('=', 1)[0].strip()
                if key_part and not key_part.replace('.', '').replace('-', '').isdigit():
                    return True
        
        # Additional check for non-ASCII metadata (common in international files)
        if self._unicode_ratio(line) > 0.3:
            # Check if it has a key-value structure even with non-ASCII characters
            if ':' in line or '=' in line:
                parts = line.split(':' if ':' in line else '=', 1)
                if len(parts) == 2:
                    key, value = parts
                    key = key.strip()
                    value = value.strip()
                    # Key should be reasonably short and not purely numeric
                    if len(key) < 50 and value and not key.replace('.', '').replace('-', '').isdigit():
                        return True
        
        return False
    
    def _detect_structure(self, lines: List[str]) -> Optional[Dict]:
        """
        Detect file structure: delimiter, header, column types.
        Based on structure_detector.py unified approach.
        """
        if not lines:
            return None
        
        delimiters = [',', '\t', ';', '|', ' ']
        best_structure = None
        best_score = 0
        
        print(f"DEBUG: Detecting structure for {len(lines)} lines")
        
        # Try each delimiter
        for delimiter in delimiters:
            delimiter_name = 'TAB' if delimiter == '\t' else f"'{delimiter}'"
            print(f"DEBUG: Trying delimiter {delimiter_name}")
            
            structure = self._analyze_delimiter_structure(lines, delimiter)
            if structure:
                print(f"DEBUG: Delimiter {delimiter_name} - Score: {structure['score']:.3f}, Cols: {structure['num_cols']}, Consistency: {structure['consistency']:.3f}")
                if structure['score'] > best_score:
                    best_score = structure['score']
                    best_structure = structure
                    print(f"DEBUG: New best delimiter: {delimiter_name} with score {structure['score']:.3f}")
            else:
                print(f"DEBUG: Delimiter {delimiter_name} - No valid structure detected")
        
        # Fallback for single-column files
        if not best_structure:
            print("DEBUG: No multi-column structure found, falling back to single-column detection")
            structure = self._analyze_single_column_structure(lines)
            if structure:
                print(f"DEBUG: Single-column structure detected - Score: {structure['score']:.3f}")
                best_structure = structure
            else:
                print("DEBUG: Single-column detection also failed")
        else:
            delimiter_name = 'TAB' if best_structure['delimiter'] == '\t' else f"'{best_structure['delimiter']}'"
            print(f"DEBUG: Final structure - Delimiter: {delimiter_name}, Strategy: {best_structure['strategy']}, Score: {best_structure['score']:.3f}")
            print(f"DEBUG: Final column types: {best_structure['column_types']}")
            if best_structure.get('headers'):
                print(f"DEBUG: Detected headers: {best_structure['headers']}")
            # Show detailed column type detection for final structure
            if best_structure['strategy'] == 'multi_column':
                print(f"DEBUG: === FINAL STRUCTURE DETAILED ANALYSIS ===")
                # Re-analyze the final structure to show detailed debug
                final_delimited_lines = [line for line in lines[:20] if best_structure['delimiter'] in line]
                if len(final_delimited_lines) >= 3:
                    final_token_matrix = []
                    for line in final_delimited_lines:
                        tokens = [token.strip() for token in line.split(best_structure['delimiter'])]
                        final_token_matrix.append(tokens)
                    
                    if final_token_matrix:
                        max_cols = max(len(row) for row in final_token_matrix)
                        for row in final_token_matrix:
                            row.extend([''] * (max_cols - len(row)))
                        
                        # Re-run column type detection with debug output
                        print(f"DEBUG: Re-analyzing final structure with delimiter {delimiter_name}")
                        self._infer_column_types(final_token_matrix)
        
        return best_structure
    
    def _analyze_delimiter_structure(self, lines: List[str], delimiter: str) -> Optional[Dict]:
        """
        Analyze structure for a specific delimiter.
        """
        # Get lines that contain the delimiter
        delimited_lines = [line for line in lines[:20] if delimiter in line]
        if len(delimited_lines) < 3:
            return None
        
        # Split into token matrix
        token_matrix = []
        for line in delimited_lines:
            tokens = [token.strip() for token in line.split(delimiter)]
            token_matrix.append(tokens)
        
        if not token_matrix:
            return None
        
        # Ensure consistent matrix shape
        max_cols = max(len(row) for row in token_matrix)
        for row in token_matrix:
            row.extend([''] * (max_cols - len(row)))
        
        # Calculate consistency score
        row_lengths = [len(row) for row in token_matrix]
        consistency = 1.0 - (np.std(row_lengths) / np.mean(row_lengths)) if np.mean(row_lengths) > 0 else 0
        
        # Detect column types
        column_types = self._infer_column_types(token_matrix)
        
        # Detect header
        header_info = self._detect_header(token_matrix, column_types)
        
        # Calculate overall score
        score = consistency * 0.6
        if 2 <= max_cols <= 20:
            score += 0.3
        if header_info.get('has_header'):
            score += 0.1
        
        return {
            'delimiter': delimiter,
            'num_cols': max_cols,
            'consistency': consistency,
            'column_types': column_types,
            'header_info': header_info,
            'headers': header_info.get('headers', []),
            'data_start_row': header_info.get('data_start_row', 0),
            'score': score,
            'strategy': 'multi_column'
        }
    
    def _analyze_single_column_structure(self, lines: List[str]) -> Optional[Dict]:
        """
        Analyze structure for single-column data.
        """
        clean_lines = [line.strip() for line in lines if line.strip()]
        if not clean_lines:
            return None
        
        # Detect if first line is a header
        has_header = False
        data_start_row = 0
        headers = []
        
        if len(clean_lines) >= 2:
            first_line = clean_lines[0]
            second_line = clean_lines[1]
            
            # Enhanced header detection for single column
            first_is_numeric = self._is_likely_numeric(first_line)
            second_is_numeric = self._is_likely_numeric(second_line)
            
            # Check if first line looks like header and second like data
            if not first_is_numeric and second_is_numeric:
                has_header = True
                data_start_row = 1  # This is correct for single column - skip first line
                headers = [first_line]
            # Additional check for non-ASCII headers
            elif not first_is_numeric and self._unicode_ratio(first_line) > 0.3:
                # Check if multiple following lines are numeric
                next_lines_numeric = 0
                for i in range(1, min(4, len(clean_lines))):
                    if self._is_likely_numeric(clean_lines[i]):
                        next_lines_numeric += 1
                
                if next_lines_numeric >= 2:  # At least 2 of next 3 lines are numeric
                    has_header = True
                    data_start_row = 1
                    headers = [first_line]
        
        # Determine column type
        data_lines = clean_lines[data_start_row:]
        if data_lines:
            column_type = self._determine_single_column_type(data_lines)
        else:
            column_type = 'unknown'
        
        return {
            'delimiter': '',
            'num_cols': 1,
            'consistency': 1.0,
            'column_types': {0: column_type},
            'header_info': {
                'has_header': has_header,
                'headers': headers,
                'data_start_row': data_start_row
            },
            'headers': headers,
            'data_start_row': data_start_row,
            'score': 0.5,
            'strategy': 'single_column'
        }
    
    def _infer_column_types(self, token_matrix: List[List[str]]) -> Dict[int, str]:
        """
        Infer column types: numerical, categorical, textual, datetime, or mixed.
        """
        if not token_matrix:
            return {}
        
        max_cols = max(len(row) for row in token_matrix)
        column_types = {}
        
        print(f"DEBUG: Inferring column types for {max_cols} columns")
        
        # Transpose matrix to get columns
        for col_idx in range(max_cols):
            column_values = []
            for row in token_matrix:
                if col_idx < len(row):
                    value = row[col_idx].strip()
                    if value and value.lower() not in {'nan', 'na', 'none', 'null', ''}:
                        column_values.append(value)
            
            if not column_values:
                column_types[col_idx] = 'unknown'
                print(f"DEBUG: Column {col_idx} - Type: unknown (no valid values)")
                continue
            
            # Check for datetime first
            if self._is_datetime_column(column_values):
                column_types[col_idx] = 'datetime'
                print(f"DEBUG: Column {col_idx} - Type: datetime (sample: {column_values[:3]})")
                continue
            
            # Check numeric ratio
            numeric_count = sum(1 for val in column_values if self._is_likely_numeric(val))
            numeric_ratio = numeric_count / len(column_values)
            
            # Check uniqueness
            unique_ratio = len(set(column_values)) / len(column_values)
            avg_length = sum(len(val) for val in column_values) / len(column_values)
            
            print(f"DEBUG: Column {col_idx} - Numeric ratio: {numeric_ratio:.3f}, Unique ratio: {unique_ratio:.3f}, Avg length: {avg_length:.1f}")
            
            # Classify column type
            if numeric_ratio > 0.8:
                column_types[col_idx] = 'numerical'
                print(f"DEBUG: Column {col_idx} - Type: numerical (sample: {column_values[:3]})")
            else:
                column_types[col_idx] = 'categorical'
                print(f"DEBUG: Column {col_idx} - Type: categorical (sample: {column_values[:3]})")
       
        
        print(f"DEBUG: Final column types: {column_types}")
        return column_types
    
    def _determine_single_column_type(self, values: List[str]) -> str:
        """
        Determine type for single column data.
        """
        if not values:
            return 'unknown'
        
        # Check for datetime
        if self._is_datetime_column(values):
            return 'datetime'
        
        # Check numeric ratio
        numeric_count = sum(1 for val in values if self._is_likely_numeric(val))
        numeric_ratio = numeric_count / len(values)
        
        # Check uniqueness
        unique_ratio = len(set(values)) / len(values)
        avg_length = sum(len(val) for val in values) / len(values)
        
        if numeric_ratio > 0.8:
            return 'numerical'
        elif unique_ratio < 0.5 and avg_length < 15 and numeric_ratio < 0.3:
            return 'categorical'
        elif avg_length > 20 or (unique_ratio > 0.9 and numeric_ratio < 0.1):
            return 'textual'
        else:
            return 'mixed'
    
    def _is_datetime_column(self, values: List[str]) -> bool:
        """
        Check if column values look like datetime data.
        """
        if len(values) < 2:
            return False
        
        # Sample a few values for testing
        sample = values[:min(5, len(values))]
        
        datetime_count = 0
        for value in sample:
            if self._is_likely_datetime(value):
                datetime_count += 1
        
        # Lower threshold for date detection to catch more cases
        return datetime_count / len(sample) > 0.5
    
    def _is_likely_datetime(self, value: str) -> bool:
        """
        Check if a value looks like datetime.
        """
        if not value or not value.strip():
            return False
            
        value = value.strip()
        
        try:
            # Try pandas datetime parsing first - this is the most reliable
            pd.to_datetime(value, errors='raise')
            return True
        except:
            # Check common datetime patterns
            datetime_patterns = [
                r'^\d{4}-\d{1,2}-\d{1,2}$',          # YYYY-MM-DD or YYYY-M-D
                r'^\d{1,2}/\d{1,2}/\d{4}$',          # MM/DD/YYYY or M/D/YYYY
                r'^\d{1,2}-\d{1,2}-\d{4}$',          # MM-DD-YYYY or M-D-YYYY
                r'^\d{4}/\d{1,2}/\d{1,2}$',          # YYYY/MM/DD or YYYY/M/D
                r'^\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}',  # YYYY-MM-DD HH:MM
                r'^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}',  # MM/DD/YYYY HH:MM
                r'^\d{4}-\d{3}$',                    # YYYY-DDD (Julian date)
                r'^\d{8}$',                          # YYYYMMDD
            ]
            
            for pattern in datetime_patterns:
                if re.match(pattern, value):
                    return True
        
        return False
    
    def _is_likely_numeric(self, value: str) -> bool:
        """
        Check if a value looks numeric.
        """
        if not value or not value.strip():
            return False
        
        try:
            float(value.strip())
            return True
        except ValueError:
            return False
    
    def _unicode_ratio(self, text: str) -> float:
        """
        Calculate the ratio of non-ASCII characters in text.
        Used for detecting non-ASCII headers in multilingual files.
        """
        return sum(1 for c in text if ord(c) > 127) / len(text) if text else 0
    
    def _detect_header(self, token_matrix: List[List[str]], column_types: Dict[int, str]) -> Dict:
        """
        Detect if there's a header row and where data starts.
        """
        if not token_matrix:
            return {'has_header': False, 'headers': [], 'data_start_row': 0}

        # Try intelligent scoring first
        for row_idx in range(min(3, len(token_matrix))):
            row = token_matrix[row_idx]
            header_score = self._calculate_header_score(row, row_idx, token_matrix, column_types)
            if header_score > 0.6:
                return {
                    'has_header': True,
                    'headers': [token.strip() for token in row],
                    'data_start_row': row_idx + 1
                }

        # Fallback: find first row that looks like a string header
        for row_idx, row in enumerate(token_matrix[:3]):
            tokens = [token.strip() for token in row if token.strip()]
            string_count = sum(1 for token in tokens if not self._is_likely_numeric(token))
            unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0
            if tokens and string_count / len(tokens) > 0.8 and unique_ratio > 0.8:
                return {
                    'has_header': True,
                    'headers': tokens,
                    'data_start_row': row_idx + 1
                }

        # Default: no header
        return {'has_header': False, 'headers': [], 'data_start_row': 0}

    
    def _calculate_header_score(self, row: List[str], row_idx: int, 
                               token_matrix: List[List[str]], column_types: Dict[int, str]) -> float:
        """
        Calculate how likely a row is to be a header.
        Enhanced for non-ASCII headers and multilingual support.
        """
        score = 0.0
        
        # Check for header keywords (reduced English bias)
        header_keywords = [
            'time', 'date', 'timestamp', 'datetime', 'seconds', 'ms', 'milliseconds',
            'signal', 'channel', 'sensor', 'value', 'data', 'measurement',
            'voltage', 'current', 'temperature', 'pressure', 'frequency',
            'x', 'y', 'z', 'index', 'sample', 'point', 'id'
        ]
        
        keyword_matches = 0
        for token in row:
            token_lower = token.strip().lower()
            if any(keyword in token_lower for keyword in header_keywords):
                keyword_matches += 1
        
        # Reduced weight for English keyword matching to reduce bias
        if keyword_matches > 0:
            score += 0.2 * (keyword_matches / len(row))  # Reduced from 0.4
        
        # Unicode ratio detection for non-ASCII headers
        unicode_ratio = np.mean([self._unicode_ratio(token) for token in row])
        if unicode_ratio > 0.5:
            score += 0.2  # Boost for non-ASCII headers
        elif unicode_ratio > 0.2:
            score += 0.1  # Smaller boost for mixed content
        
        # Check if non-numeric in numeric columns
        numeric_mismatches = 0
        for col_idx, token in enumerate(row):
            if col_idx in column_types and column_types[col_idx] == 'numerical':
                if not self._is_likely_numeric(token):
                    numeric_mismatches += 1
        
        numeric_cols = sum(1 for col_type in column_types.values() if col_type == 'numerical')
        if numeric_cols > 0:
            score += 0.3 * (numeric_mismatches / numeric_cols)
        
        # All-text vs numeric-next-row bonus
        current_row_numeric_count = sum(1 for token in row if self._is_likely_numeric(token))
        current_row_numeric_ratio = current_row_numeric_count / len(row) if row else 0
        
        # If current row is entirely non-numeric, check if next rows are numeric
        if current_row_numeric_ratio == 0 and row_idx < len(token_matrix) - 1:
            next_rows_numeric_ratios = []
            for next_row_idx in range(row_idx + 1, min(row_idx + 4, len(token_matrix))):  # Check next 3 rows
                next_row = token_matrix[next_row_idx]
                if next_row:
                    next_numeric_count = sum(1 for token in next_row if self._is_likely_numeric(token))
                    next_numeric_ratio = next_numeric_count / len(next_row)
                    next_rows_numeric_ratios.append(next_numeric_ratio)
            
            if next_rows_numeric_ratios:
                avg_next_numeric_ratio = np.mean(next_rows_numeric_ratios)
                if avg_next_numeric_ratio > 0.7:  # Next rows are mostly numeric
                    score += 0.25  # Strong boost for all-text header followed by numeric data
                elif avg_next_numeric_ratio > 0.4:
                    score += 0.15  # Moderate boost
        
        # Check uniqueness compared to following rows
        if row_idx < len(token_matrix) - 1:
            uniqueness = self._calculate_row_uniqueness(row, token_matrix[row_idx+1:])
            score += 0.2 * uniqueness  # Reduced from 0.3 to balance with new features
        
        return min(score, 1.0)
    
    def _calculate_row_uniqueness(self, row: List[str], following_rows: List[List[str]]) -> float:
        """
        Calculate how unique a row is compared to following rows.
        """
        if not following_rows:
            return 0.0
        
        unique_tokens = 0
        for col_idx, token in enumerate(row):
            is_unique = True
            for other_row in following_rows[:5]:  # Check first 5 following rows
                if col_idx < len(other_row) and other_row[col_idx].strip().lower() == token.strip().lower():
                    is_unique = False
                    break
            if is_unique and token.strip():
                unique_tokens += 1
        
        return unique_tokens / len(row) if row else 0.0
    
    def _parse_data(self, lines: List[str], structure_info: Dict) -> Optional[pd.DataFrame]:
        """
        Parse data into DataFrame based on detected structure.
        """
        try:
            delimiter = structure_info['delimiter']
            headers = structure_info.get('headers', [])
            data_start_row = structure_info.get('data_start_row', 0)
            
            # Skip header rows to get to data
            data_lines = lines[data_start_row:]
            if not data_lines:
                return None
            
            if delimiter == '':
                # Single column data
                if headers:
                    df = pd.DataFrame({headers[0]: [line.strip() for line in data_lines]})
                else:
                    df = pd.DataFrame({'Column_0': [line.strip() for line in data_lines]})
            else:
                # Multi-column data
                data_content = '\n'.join(data_lines)
                
                try:
                    if headers:
                        df = pd.read_csv(
                            pd.io.common.StringIO(data_content),
                            sep=delimiter,
                            names=headers,
                            header=None,
                            engine='python',
                            on_bad_lines='skip',
                            encoding_errors='ignore'
                        )
                    else:
                        df = pd.read_csv(
                            pd.io.common.StringIO(data_content),
                            sep=delimiter,
                            header=None,
                            engine='python',
                            on_bad_lines='skip',
                            encoding_errors='ignore'
                        )
                except:
                    # Fallback manual parsing
                    df = self._manual_parse(data_lines, delimiter, headers)
            
            if df is None or df.empty:
                return None
            
            # Convert columns to appropriate types
            df = self._convert_column_types(df, structure_info.get('column_types', {}))
            
            return df
            
        except Exception as e:
            print(f"Error parsing data: {e}")
            return None
    
    def _manual_parse(self, lines: List[str], delimiter: str, headers: List[str]) -> Optional[pd.DataFrame]:
        """
        Manual parsing fallback.
        """
        try:
            rows = []
            for line in lines:
                if line.strip():
                    row = [cell.strip() for cell in line.split(delimiter)]
                    rows.append(row)
            
            if not rows:
                return None
            
            # Ensure consistent row length
            max_cols = max(len(row) for row in rows)
            for row in rows:
                row.extend([''] * (max_cols - len(row)))
            
            # Create DataFrame
            if headers and len(headers) == max_cols:
                df = pd.DataFrame(rows, columns=headers)
            else:
                df = pd.DataFrame(rows)
            
            return df
            
        except Exception:
            return None
    
    def _convert_column_types(self, df: pd.DataFrame, column_types: Dict[int, str]) -> pd.DataFrame:
        """
        Convert DataFrame columns to appropriate types.
        """
        # Initialize category mappings storage if not exists
        if not hasattr(df, '_category_mappings'):
            df._category_mappings = {}
        
        for col_idx, col in enumerate(df.columns):
            col_type = column_types.get(col_idx, 'unknown')
            
            if col_type == 'datetime':
                try:
                    # Convert to datetime and store both datetime and numeric versions
                    datetime_series = pd.to_datetime(df[col], errors='coerce')
                    if not datetime_series.isnull().all():
                        # Store original datetime values as metadata
                        df[f'{col}_datetime'] = datetime_series
                        # Convert to numeric for plotting (seconds since first date)
                        first_date = datetime_series.dropna().iloc[0]
                        df[col] = (datetime_series - first_date).dt.total_seconds()  # Seconds
                    else:
                        # Fallback to numeric if datetime conversion fails
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                except Exception:
                    # Fallback to numeric
                    df[col] = pd.to_numeric(df[col], errors='ignore')
            elif col_type == 'numerical':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif col_type == 'categorical':
                try:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    # Ensure strings, fill NaNs to avoid crashing
                    filled = df[col].astype(str).fillna('__missing__')
                    df[col] = le.fit_transform(filled)
                    df[col] = df[col].astype(float)
                    
                    # Save the category mapping
                    category_mapping = {idx: label for idx, label in enumerate(le.classes_)}
                    df._category_mappings[col] = category_mapping
                    
                    # Debug output for category mapping
                    print(f"DEBUG: Category mapping for column '{col}': {category_mapping}")
                    
                except Exception as e:
                    print(f"Warning: Failed to encode column '{col}': {e}")
            else:
                # Try numeric conversion anyway
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if not numeric_series.isnull().all():
                    df[col] = numeric_series
        
        # Debug output for all category mappings
        if df._category_mappings:
            print(f"DEBUG: All category mappings: {df._category_mappings}")
        
        return df
    
    def _detect_x_column(self, df: pd.DataFrame, headers: List[str]) -> Optional[Dict]:
        """
        Detect the X (time) column with priority for time/date substrings.
        """
        if df.empty or len(df.columns) < 2:
            return None
        
        candidates = []
        
        # Strategy 1: Exact time name matching (absolute highest priority)
        exact_time_names = ['time', 'timestamp', 'datetime', 'date']
        for col in df.columns:
            col_str = str(col).lower().strip()
            if col_str in exact_time_names:
                # Absolute priority for exact matches
                return {'column_name': col, 'score': 1000, 'method': 'exact_time_name'}
        
        # Strategy 2: Header name analysis with time indicators
        time_indicators = [
            'time', 'timestamp', 'datetime', 'date', 'seconds', 'ms', 'milliseconds',
            'sec', 'second', 'minute', 'hour', 'epoch', 'utc', 'gmt', 'clock'
        ]
        
        for col in df.columns:
            col_str = str(col).lower().strip()
            score = 0
            
            # Check for time indicators in column name (very high priority)
            for indicator in time_indicators:
                if indicator in col_str:
                    score += 500  # Much higher than before
                    break
            
            # Common time patterns
            if any(pattern in col_str for pattern in ['time(s)', 'time_', '_time', 't_']):
                score += 450
            elif col_str in ['t', 'x', 'index', 'idx', 'sample', 'n']:
                score += 200
            
            # Safe first column bonus with conditions
            if col == df.columns[0]:
                # Only apply bonus for multi-column files
                if len(df.columns) > 1:
                    # Don't apply if already high score from name matching (500+)
                    if score < 500:
                        # Don't apply if column appears to be categorical
                        if not self._is_categorical_column(df[col]):
                            score += 15
            
            if score > 0:
                candidates.append({'column_name': col, 'score': score, 'method': 'header_name'})
        
        # Strategy 3: Datetime data detection (lower priority than header names)
        for col in df.columns:
            if not any(c['column_name'] == col for c in candidates):
                if self._could_be_datetime_data(df[col]):
                    score = 120
                    
                    # Safe first column bonus with conditions
                    if col == df.columns[0]:
                        # Only apply bonus for multi-column files
                        if len(df.columns) > 1:
                            # Don't apply if already high score
                            if score < 500:
                                # Don't apply if column appears to be categorical
                                if not self._is_categorical_column(df[col]):
                                    score += 15
                    
                    candidates.append({'column_name': col, 'score': score, 'method': 'datetime_data'})
        
        # Strategy 4: Numeric time-like properties (lowest priority)
        for col in df.columns:
            if not any(c['column_name'] == col for c in candidates):
                if pd.api.types.is_numeric_dtype(df[col]):
                    time_score = self._score_time_like_numeric(df[col])
                    if time_score > 40:
                        # Cap the score so it can't beat header name detection
                        time_score = min(time_score, 100)
                        
                        # Safe first column bonus with conditions
                        if col == df.columns[0]:
                            # Only apply bonus for multi-column files
                            if len(df.columns) > 1:
                                # Don't apply if already high score
                                if time_score < 500:
                                    # Don't apply if column appears to be categorical
                                    if not self._is_categorical_column(df[col]):
                                        time_score += 15
                        
                        candidates.append({'column_name': col, 'score': time_score, 'method': 'numeric_time'})
        
        # Select best candidate
        if candidates:
            best_candidate = max(candidates, key=lambda x: x['score'])
            if best_candidate['score'] > 50:
                return best_candidate
        
        return None
    
    def _could_be_datetime_data(self, series: pd.Series) -> bool:
        """
        Check if series contains datetime-like data.
        """
        if len(series) < 2:
            return False
        
        # Check if already converted to timestamp
        if pd.api.types.is_numeric_dtype(series):
            # Large numbers could be timestamps
            values = series.dropna()
            if len(values) > 0:
                mean_val = values.mean()
                # Unix timestamp range check (1970-2100)
                if 0 < mean_val < 4e9:
                    return True
        
        return False
    
    def _score_time_like_numeric(self, series: pd.Series) -> float:
        """
        Score numeric series for time-like characteristics.
        """
        values = series.dropna().values
        if len(values) < 3:
            return 0
        
        score = 0
        
        # Monotonic increasing
        if np.all(np.diff(values) >= 0):
            score += 50
        
        # Starts near zero
        if len(values) > 0 and abs(values[0]) < max(1.0, np.mean(values) * 0.1):
            score += 25
        
        # Regular spacing
        if len(values) > 2:
            diffs = np.diff(values)
            if len(diffs) > 1:
                cv = np.std(diffs) / np.mean(diffs) if np.mean(diffs) > 0 else float('inf')
                if cv < 0.05:
                    score += 30
                elif cv < 0.2:
                    score += 15
        
        return score
    
    def _create_channels(self, df: pd.DataFrame, file_obj: File, 
                        x_col_info: Optional[Dict], structure_info: Dict) -> List[Channel]:
        """
        Create Channel objects from DataFrame.
        """
        channels = []
        
        try:
            # Determine X-axis data
            if x_col_info:
                x_col_name = x_col_info['column_name']
                xdata = df[x_col_name].values
                xlabel = str(x_col_name)
                
                # Check if this is a datetime column (has corresponding _datetime column)
                datetime_col_name = f'{x_col_name}_datetime'
                if datetime_col_name in df.columns:
                    xlabel = f"{x_col_name} (seconds from {df[datetime_col_name].dropna().iloc[0].strftime('%Y-%m-%d %H:%M:%S')})"
                elif x_col_info.get('method') in ['datetime_data', 'exact_time_name']:
                    xlabel = f"{x_col_name} (datetime)"
            else:
                x_col_name = None
                xdata = np.arange(len(df))
                xlabel = "Index"
            
            # Get Y columns (excluding X column)
            y_columns = [col for col in df.columns if col != x_col_name]
            
            # Filter to numeric columns for signal data
            numeric_y_cols = []
            for col in y_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Check if mostly non-null
                    non_null_ratio = df[col].notna().sum() / len(df[col])
                    if non_null_ratio > 0.1:  # At least 10% valid data
                        numeric_y_cols.append(col)
            
            # Handle single column case
            if len(df.columns) == 1:
                single_col = df.columns[0]
                if pd.api.types.is_numeric_dtype(df[single_col]):
                    channel = Channel.from_parsing(
                        file_id=file_obj.file_id,
                        filename=file_obj.filename,
                        xdata=np.arange(len(df)),
                        ydata=df[single_col].values,
                        xlabel="Index",
                        ylabel=str(single_col),
                        legend_label=str(single_col),  # Just use the header name
                        metadata={'original_column': single_col, 'parse_method': 'single_column'}
                    )
                    return [channel]
            
            # Create channels for Y columns
            if not numeric_y_cols:
                print(f"No numeric signal columns found in {file_obj.filename}")
                return []
            
            for col in numeric_y_cols:
                ydata = df[col].values
                ylabel = str(col)
                
                metadata = {
                    'original_column': col,
                    'parse_method': 'auto_robust',
                    'x_column': x_col_name if x_col_name else 'index',
                    'x_detection_method': x_col_info.get('method', 'none') if x_col_info else 'none',
                    'data_quality': 'high' if pd.Series(ydata).notna().sum() / len(ydata) > 0.9 else 'medium'
                }

                # Add category mapping if present
                if hasattr(df, '_category_mappings') and col in df._category_mappings:
                    metadata['category_mapping'] = df._category_mappings[col]
                    print(f"DEBUG: Added category mapping to channel metadata for column '{col}': {df._category_mappings[col]}")
                
                # Add datetime information if X column is datetime
                if x_col_info and f'{x_col_name}_datetime' in df.columns:
                    metadata['x_is_datetime'] = True
                    metadata['x_datetime_column'] = f'{x_col_name}_datetime'
                    metadata['x_datetime_first'] = df[f'{x_col_name}_datetime'].dropna().iloc[0].isoformat()
                    
                    # Compute datetime-based sampling rate
                    datetime_sampling_stats = self._compute_datetime_sampling_rate(df[f'{x_col_name}_datetime'])
                    if datetime_sampling_stats:
                        metadata['datetime_sampling_stats'] = datetime_sampling_stats
                else:
                    metadata['x_is_datetime'] = False
                
                channel = Channel.from_parsing(
                    file_id=file_obj.file_id,
                    filename=file_obj.filename,
                    xdata=xdata.copy(),
                    ydata=ydata,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    legend_label=str(ylabel),  # Just use the header name
                    metadata=metadata
                )
                
                channels.append(channel)
            
            return channels
            
        except Exception as e:
            print(f"Error creating channels: {e}")
            return []
    
    def _compute_datetime_sampling_rate(self, datetime_series: pd.Series) -> Optional[Dict]:
        """
        Compute sampling rate statistics from datetime series.
        
        Args:
            datetime_series: pandas Series with datetime values
            
        Returns:
            Dictionary with sampling rate statistics or None if computation fails
        """
        try:
            # Remove any NaN values and ensure we have enough data points
            clean_series = datetime_series.dropna()
            if len(clean_series) < 2:
                return None
            
            # Calculate time differences in seconds
            time_diffs = clean_series.diff().dropna()
            time_diffs_seconds = time_diffs.dt.total_seconds()
            
            # Filter out zero or negative differences
            valid_diffs = time_diffs_seconds[time_diffs_seconds > 0]
            if len(valid_diffs) == 0:
                return None
            
            # Convert to sampling rates (Hz)
            sampling_rates = 1.0 / valid_diffs
            
            # Calculate statistics
            median_fs = float(np.median(sampling_rates))
            std_fs = float(np.std(sampling_rates))
            min_fs = float(np.min(sampling_rates))
            max_fs = float(np.max(sampling_rates))
            
            # Calculate regularity score (coefficient of variation)
            cv = std_fs / median_fs if median_fs > 0 else float('inf')
            regularity_score = max(0.0, min(1.0, 1.0 - cv))
            
            # Determine time unit and create human-readable description
            median_interval_seconds = 1.0 / median_fs
            
            if median_interval_seconds < 1:
                interval_desc = f"{median_interval_seconds*1000:.1f} ms"
            elif median_interval_seconds < 60:
                interval_desc = f"{median_interval_seconds:.1f} s"
            elif median_interval_seconds < 3600:
                interval_desc = f"{median_interval_seconds/60:.1f} min"
            else:
                interval_desc = f"{median_interval_seconds/3600:.1f} h"
            
            return {
                'median_fs': median_fs,
                'std_fs': std_fs,
                'min_fs': min_fs,
                'max_fs': max_fs,
                'regularity_score': regularity_score,
                'interval_description': interval_desc,
                'total_duration_seconds': (clean_series.iloc[-1] - clean_series.iloc[0]).total_seconds(),
                'sample_count': len(clean_series),
                'method': 'datetime_based'
            }
            
        except Exception as e:
            print(f"Error computing datetime sampling rate: {e}")
            return None
    
    def _is_categorical_column(self, series: pd.Series) -> bool:
        """
        Determine if a numeric column represents categorical data.
        """
        if not pd.api.types.is_numeric_dtype(series):
            return True  # Non-numeric is likely categorical
        
        # Check for small number of unique values
        unique_count = series.nunique()
        total_count = len(series.dropna())
        
        if total_count == 0:
            return False
        
        unique_ratio = unique_count / total_count
        
        # If very few unique values relative to total, likely categorical
        if unique_ratio < 0.1 and unique_count < 20:
            return True
        
        # Check if values are all integers (could be category codes)
        if series.dropna().apply(lambda x: float(x).is_integer()).all():
            if unique_count < 50:  # Reasonable number of categories
                return True
        
        return False
    
    def _determine_data_type(self, series: pd.Series) -> DataType:
        """
        Determine DataType enum for a series.
        """
        if pd.api.types.is_numeric_dtype(series):
            if self._is_categorical_column(series):
                return DataType.TEXT  # Categorical treated as text for now
            else:
                return DataType.NUMERIC
        else:
            return DataType.TEXT
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parsing statistics."""
        stats = self.parse_stats.copy()
        
        # Add average parse time
        if self.parse_stats['files_parsed'] > 0:
            stats['avg_parse_time_ms'] = self.parse_stats['parse_time_ms'] / self.parse_stats['files_parsed']
        
        # Convert Counter to dict for JSON serialization
        stats['strategies_used'] = dict(stats['strategies_used'])
        
        return stats
    
    def reset_stats(self):
        """Reset parsing statistics."""
        self.parse_stats = {
            'files_parsed': 0,
            'parse_time_ms': 0,
            'metadata_lines_skipped': 0,
            'strategies_used': Counter()
        } 