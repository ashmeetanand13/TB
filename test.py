import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from scipy import stats


def process_data(data_text, header_row_index=3, data_start_row_index=5, column_renames=None):
    """
    Process raw temperature data from the text file.
    
    Args:
        data_text (str): Raw text data from the file
        header_row_index (int): Row index to use as header (0-based)
        data_start_row_index (int): Row index to start data from (0-based)
        column_renames (dict): Dictionary mapping original column names to new names
        
    Returns:
        tuple: (DataFrame of processed data, experiment info string)
    """
    # Split the data into lines
    lines = data_text.strip().split('\n')
    
    # Check if data exists
    if not lines:
        return pd.DataFrame(), "No data found"
    
    # Extract experiment info from the first line
    experiment_info = lines[0] if lines else "Temperature Analysis"
    
    # Determine if this is a comma-separated or space-separated file
    is_csv = ',' in lines[0] if lines else False
    
    st.write(f"Detected format: {'CSV' if is_csv else 'Space-separated'}")
    
    # Try to automatically detect the structure
    # First, check if we can use pandas read_csv directly
    try:
        # Try to read with pandas - may work for standard CSV files
        buffer = io.StringIO(data_text)
        df = pd.read_csv(buffer, sep=None, engine='python', skiprows=header_row_index)
        st.write("Successfully read file with pandas auto-detection")
        
        # If dataframe has only one column, it might be space-delimited incorrectly identified as one column
        if df.shape[1] == 1:
            st.warning("File detected as single column - trying different approach")
            raise ValueError("Single column detected")
            
        # Print shape for debugging
        st.write(f"DataFrame shape: {df.shape}")
        
        # Display first few rows for debugging
        st.write("First few rows of raw data:")
        st.write(df.head())
        
        # Rename columns if provided
        if column_renames and isinstance(column_renames, dict):
            df = df.rename(columns=column_renames)
            st.write("Columns renamed successfully!")
            st.write(df.head())
        
        return df, experiment_info
        
    except Exception as e:
        st.warning(f"Could not automatically read file: {str(e)}")
        st.write("Falling back to manual parsing...")
    
    # Manual parsing approach
    # Extract the header (column names)
    if len(lines) <= header_row_index:
        return pd.DataFrame(), "Header row not found"
        
    header_line = lines[header_row_index]
    
    # Create column names
    if is_csv:
        # For CSV files
        columns = [col.strip() for col in header_line.split(',') if col.strip()]
    else:
        # For space-delimited files, try to split intelligently
        # This is a bit tricky - let's look for common patterns in temperature data
        
        # Method 1: Split by multiple spaces (common in fixed-width formats)
        columns = [col.strip() for col in header_line.split('  ') if col.strip()]
        
        # If that produces too few columns, try regular split
        if len(columns) < 4:
            columns = [col.strip() for col in header_line.split() if col.strip()]
    
    st.write(f"Detected {len(columns)} columns: {columns}")
    
    # If we have a very small number of columns, try to parse as fixed width
    if len(columns) < 4:
        st.warning("Very few columns detected. Trying fixed-width parsing.")
        try:
            buffer = io.StringIO(data_text)
            # Try pandas fixed width parser
            df = pd.read_fwf(buffer, skiprows=header_row_index)
            st.write(f"Fixed-width parsing successful. Shape: {df.shape}")
            
            # Display first few rows for debugging
            st.write("First few rows of raw data:")
            st.write(df.head())
            
            # Rename columns if provided
            if column_renames and isinstance(column_renames, dict):
                df = df.rename(columns=column_renames)
                st.write("Columns renamed successfully!")
                st.write(df.head())
            
            return df, experiment_info
        except Exception as e:
            st.error(f"Fixed-width parsing failed: {str(e)}")
    
    # Extract data rows starting from the data_start_row_index
    data_rows = []
    for line in lines[data_start_row_index:]:
        if not line.strip():  # Skip empty lines
            continue
            
        if is_csv:
            # For CSV files
            values = [val.strip() for val in line.split(',')]
        else:
            # For space-delimited files
            values = [val.strip() for val in line.split()]
            
        if values:
            data_rows.append(values)
    
    # Check if we have any data rows
    if not data_rows:
        return pd.DataFrame(), "No valid data rows found"
    
    # Create DataFrame with the rows we found
    df = pd.DataFrame(data_rows)
    
    # Print shape for debugging
    st.write(f"DataFrame shape: {df.shape}")
    
    # Create proper column names for the data
    # If we have more columns than headers, add dummy headers
    if df.shape[1] > len(columns):
        for i in range(len(columns), df.shape[1]):
            columns.append(f"Column_{i+1}")
    
    # If we have fewer columns than expected, use what we have
    actual_columns = columns[:df.shape[1]]
    
    # Make sure we don't have duplicate column names
    col_counter = {}
    unique_columns = []
    
    for col in actual_columns:
        if col in col_counter:
            col_counter[col] += 1
            unique_columns.append(f"{col}_{col_counter[col]}")
        else:
            col_counter[col] = 0
            unique_columns.append(col)
    
    # Set unique column names
    df.columns = unique_columns
    
    # Apply column renaming if provided
    if column_renames and isinstance(column_renames, dict):
        df = df.rename(columns=column_renames)
        st.write("Columns renamed successfully!")
    
    # Display first few rows for debugging
    st.write("First few rows of raw data:")
    st.write(df.head())
    
    # Display first few rows before conversion
    st.write("Raw data before conversion:")
    st.write(df.head())
    
    # Convert all columns to numeric
    for col in df.columns:
        # Skip any columns that look like they're not meant to be numeric (e.g., labels, dates)
        if col in ['Time', 'Date', 'Label', 'Description']:
            continue
            
        # Replace known non-numeric values
        df[col] = df[col].replace(['High', 'INF', '#INF', 'NaN', '-INF', '#VALUE!', '°C', '########## '], float('nan'))
        
        # Try to convert to numeric
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            st.warning(f"Could not convert column {col} to numeric: {str(e)}")
    
    # Report on NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        st.warning(f"Some values couldn't be converted to numbers. NaN count: {nan_count}")
        
        # Show which columns have NaNs
        nan_cols = df.columns[df.isna().any()].tolist()
        st.write(f"Columns with NaN values: {nan_cols}")
    
    # Keep a copy of data with NaNs for visualization purposes
    df_with_nans = df.copy()
    
    # Option to keep rows with NaN values
    keep_nan_rows = st.sidebar.checkbox("Keep rows with non-numeric values", value=False)
    
    if keep_nan_rows:
        # For calculations, replace NaNs with the column mean
        for col in df.columns:
            if df[col].isna().any() and not df[col].isna().all():
                try:
                    col_mean = df[col].mean()
                    df[col] = df[col].fillna(col_mean)
                    st.info(f"NaN values in {col} were replaced with the column mean: {col_mean:.2f}")
                except:
                    st.warning(f"Could not calculate mean for column {col}")
        return df, experiment_info
    else:
        # Remove rows with NaN values
        df_cleaned = df.dropna()
        
        # Check if we lost too many rows
        rows_lost = len(df) - len(df_cleaned)
        if rows_lost > 0:
            st.warning(f"Removed {rows_lost} row(s) due to non-numeric values.")
        
        if df_cleaned.empty:
            st.warning("All rows contained non-numeric values. Using data with NaNs.")
            return df_with_nans, f"{experiment_info} (using data with NaNs)"
        
        return df_cleaned, experiment_info

def merge_selected_files(processed_files, file_selections, column_selections):
    """
    Merge selected columns from multiple files into a single DataFrame.
    
    Args:
        processed_files (dict): Dictionary of processed file data
        file_selections (list): List of selected file names to merge
        column_selections (dict): Dictionary mapping file names to selected columns
        
    Returns:
        tuple: (DataFrame of merged data, combined experiment info string)
    """
    
    if not file_selections:
        st.error("No files selected for merging")
        return pd.DataFrame(), "No files selected for merging"
    
    # Create a new empty DataFrame for the merged result
    merged_df = pd.DataFrame()
    experiment_infos = []
    
    # Debug information
    st.write("Selected files for merging:", file_selections)
    total_selected_columns = sum(len(column_selections.get(file, [])) for file in file_selections)
    st.write(f"Total selected columns across all files: {total_selected_columns}")
    
    # Track the longest DataFrame to ensure all have same length
    max_length = 0
    longest_df = None
    
    # First pass: find the longest DataFrame
    for file_name in file_selections:
        if file_name in processed_files:
            file_data = processed_files[file_name]
            df = file_data['df']
            # Get length of this DataFrame
            if len(df) > max_length:
                max_length = len(df)
                longest_df = df
                st.write(f"Longest DataFrame: {file_name} with {max_length} rows")
    
    if longest_df is None:
        st.error("No valid data found in selected files")
        return pd.DataFrame(), "No valid data found in selected files"
    
    # Create time/sample column for the merged data
    # Look for existing time/sample column in the longest DataFrame
    sample_col = None
    for col in longest_df.columns:
        col_lower = str(col).lower()
        if 'sample' in col_lower or 'time' in col_lower or 'second' in col_lower:
            sample_col = col
            break
    
    # If found, add it to the merged DataFrame
    if sample_col is not None:
        st.write(f"Using existing sample column: {sample_col}")
        merged_df['Sample'] = pd.to_numeric(longest_df[sample_col], errors='coerce')
    else:
        # Create a sample column based on index
        st.write("Creating new sample column based on index")
        merged_df['Sample'] = range(max_length)
    
    # Second pass: merge data
    for file_name in file_selections:
        if file_name in processed_files:
            file_data = processed_files[file_name]
            df = file_data['df']
            exp_info = file_data['experiment_info']
            experiment_infos.append(exp_info)
            
            # Get selected columns for this file
            selected_cols = column_selections.get(file_name, [])
            st.write(f"Selected columns for {file_name}: {selected_cols}")
            
            if not selected_cols:
                st.warning(f"No columns selected for {file_name}")
                continue
            
            # Check if the DataFrame has any of the selected columns
            existing_cols = [col for col in selected_cols if col in df.columns]
            if not existing_cols:
                st.warning(f"None of the selected columns exist in {file_name}")
                continue
            
            st.write(f"Found columns in {file_name}: {existing_cols}")
            
            # Handle length differences - pad shorter DataFrames with NaN
            if len(df) < max_length:
                # Create empty rows to match lengths
                pad_length = max_length - len(df)
                st.write(f"Padding {file_name} with {pad_length} rows")
                pad_df = pd.DataFrame(np.nan, index=range(pad_length), columns=df.columns)
                df = pd.concat([df, pad_df], ignore_index=True)
            elif len(df) > max_length:
                # Truncate longer DataFrames
                st.write(f"Truncating {file_name} to {max_length} rows")
                df = df.iloc[:max_length]
            
            # Add columns to merged DataFrame with prefix to avoid naming conflicts
            for col in existing_cols:
                # Create a unique column name by prefixing with shortened file name
                # Extract just filename without extension
                file_prefix = file_name.split('.')[0]
                if len(file_prefix) > 10:
                    file_prefix = file_prefix[:10]
                new_col_name = f"{file_prefix}_{col}"
                
                # Add to merged DataFrame and convert to numeric
                try:
                    # Make sure to convert to numeric
                    merged_df[new_col_name] = pd.to_numeric(df[col], errors='coerce')
                    st.write(f"Added column {new_col_name} with {merged_df[new_col_name].count()} non-NaN values")
                except Exception as e:
                    st.error(f"Error adding column {col} from {file_name}: {str(e)}")
    
    # Create combined experiment info
    combined_info = "Combined data from: " + ", ".join(experiment_infos)
    
    # Final check - make sure we have more than just the Sample column
    if merged_df.shape[1] <= 1:
        st.error("Failed to create merged dataset - no data columns were added")
        return pd.DataFrame(), "No valid data columns were added"
    
    # Check for valid numeric data
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'Sample']
    
    if not numeric_cols:
        st.error("No valid numeric data columns in combined dataset")
        return pd.DataFrame(), "No valid numeric data columns"
    
    st.write(f"Successfully created combined dataset with {len(numeric_cols)} numeric columns")
    st.write("Preview of combined data:")
    st.write(merged_df.head())
    
    return merged_df, combined_info

def find_time_to_reach(df, column, threshold, is_heat_analysis=True):
    """
    Find the time (sample) when a temperature threshold is first reached.
    
    Args:
        df (DataFrame): Processed temperature data
        column (str): Column name to analyze
        threshold (float): Temperature threshold to look for
        is_heat_analysis (bool): If True, find when temperature rises above threshold.
                                If False, find when temperature falls below threshold.
        
    Returns:
        str: Formatted time string or "N/A" if threshold is never reached
    """
    # Check if column exists in DataFrame
    if column not in df.columns:
        return "Column not found"
        
    # Find the first time the temperature crosses the threshold
    try:
        # Make sure the column contains numeric data
        if pd.api.types.is_numeric_dtype(df[column]):
            if is_heat_analysis:
                # For heat analysis: when temperature first goes above threshold
                matching_rows = df[df[column] >= threshold]
            else:
                # For cold analysis: when temperature first goes below threshold
                matching_rows = df[df[column] <= threshold]
            
            if not matching_rows.empty:
                # Get the first sample (row index) where threshold is reached
                first_index = matching_rows.index[0]
                
                # Look for a Sample No or similar column to get actual time
                sample_col = None
                for col in df.columns:
                    if 'sample' in str(col).lower() or 'time' in str(col).lower() or 'second' in str(col).lower():
                        sample_col = col
                        break
                
                # Fix indentation issue here - these statements should be outside the for loop
                if sample_col is not None and pd.api.types.is_numeric_dtype(df[sample_col]):
                    # Use the actual sample number from the data
                    time_seconds = float(df.iloc[first_index][sample_col])
                else:
                    # Time in seconds is just the row index (assuming 1 Hz sampling)
                    time_seconds = first_index
                
                # Format as time (MM:SS)
                if pd.isna(time_seconds):
                    return f"N/A (sample {first_index})"
                else:
                    minutes = int(time_seconds) // 60
                    seconds = int(time_seconds) % 60
                    return f"{minutes}:{seconds:02d} (sample {first_index})"
               
            return "N/A"  # If threshold is never reached
        else:
            return "Non-numeric column"
    except Exception as e:
        return f"Error: {str(e)}"


def calculate_time_threshold(df, column, threshold, is_heat_analysis=True):
    """
    Calculate the time spent above/below a specific temperature threshold.
    
    Args:
        df (DataFrame): Processed temperature data
        column (str): Column name to analyze
        threshold (float): Temperature threshold to look for
        is_heat_analysis (bool): If True, calculate time above threshold.
                                If False, calculate time below threshold.
        
    Returns:
        tuple: (Total time spent above/below threshold, Percentage of total time)
    """
    # Check if column exists in DataFrame
    if column not in df.columns:
        return "Column not found", "N/A"
        
    try:
        # Make sure the column contains numeric data
        if pd.api.types.is_numeric_dtype(df[column]):
            # Count samples above/below threshold
            if is_heat_analysis:
                matching_samples = df[df[column] >= threshold]
            else:
                matching_samples = df[df[column] <= threshold]
                
            total_samples = len(df)
            samples_count = len(matching_samples)
            
            if samples_count == 0:
                return "0:00 (0 samples)", "0.00%"
            
            # Calculate percentage
            percentage = (samples_count / total_samples) * 100
            
            # Convert to time format (assuming 1 second per sample)
            minutes = samples_count // 60
            seconds = samples_count % 60
            
            return f"{minutes}:{seconds:02d} ({samples_count} samples)", f"{percentage:.2f}%"
        else:
            return "Non-numeric column", "N/A"
    except Exception as e:
        return f"Error: {str(e)}", "N/A"


def calculate_statistics(df, is_heat_analysis=True):
    """
    Calculate statistics for temperature channels.
    
    Args:
        df (DataFrame): Processed temperature data
        is_heat_analysis (bool): If True, calculate heat-related metrics.
                               If False, calculate cold-related metrics.
        
    Returns:
        dict: Dictionary of statistics for each channel
    """
    stats_results = {}
    
    # Check if DataFrame is empty
    if df.empty:
        st.error("No data to analyze. DataFrame is empty.")
        return {}
        
    # Get temperature columns from session state if available
    temp_columns = st.session_state.get('temp_columns', [])
    
    # If this is a combined dataset, check for combined columns
    is_combined = "Combined_Data" in st.session_state.get('selected_file', '')
    if is_combined:
        combined_columns = st.session_state.get('combined_temp_columns', [])
        if combined_columns:
            temp_columns = combined_columns
            st.write(f"Using combined data columns: {temp_columns}")
    
    # If no columns were set, try to find some reasonable defaults
    if not temp_columns:
        # Get temperature column names - REMOVED the 4-channel limit
        temp_columns = []
        
        # First, try to find columns with expected temperature names
        for col in df.columns:
            col_str = str(col).lower()
            # Look for columns with temperature-related names
            if ('channel' in col_str) or ('temp' in col_str) or ('°c' in col_str) or ('temperature' in col_str):
                # Make sure it's numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    temp_columns.append(col)
        
        # If we couldn't find any columns that way, use all numeric columns
        if not temp_columns:
            st.warning("No specific temperature columns found. Using all numeric columns.")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                # Skip columns that look like sample numbers or time
                col_str = str(col).lower()
                if 'sample' in col_str or 'time' in col_str or 'second' in col_str or 'minute' in col_str:
                    continue
                temp_columns.append(col)
    
    st.write(f"Using channels for statistics: {temp_columns}")
    
    # Store the columns for use in other functions
    st.session_state['temp_columns'] = temp_columns
    
    # Calculate statistics for each column
    for col in temp_columns:
        if col not in df.columns:
            continue
            
        # Check if column has any non-NaN values
        if df[col].isna().all():
            continue
            
        # Basic statistics
        stats_results[col] = {
            'Minimum': df[col].min(),
            'Maximum': df[col].max(),
            'Average': df[col].mean(),
            'Standard deviation': df[col].std()
        }
        
        # Safely calculate mode (might fail if all values are unique)
        try:
            stats_results[col]['Mode'] = stats.mode(df[col].dropna(), keepdims=False)[0]
        except:
            most_common = df[col].value_counts().index[0] if not df[col].empty else None
            stats_results[col]['Mode'] = most_common
        
        # Define thresholds for temperature goals
        thresholds = st.session_state.get('threshold_list', [38,41,43])
        
        # Calculate time to reach each threshold
        for temp in thresholds:
            # For heat analysis: time to reach rising temperature
            # For cold analysis: time to reach falling temperature
            result = find_time_to_reach(df, col, temp, is_heat_analysis)
            
            if is_heat_analysis:
                stats_results[col][f'Time to reach {temp}°C'] = result
            else:
                stats_results[col][f'Time to reach below {temp}°C'] = result
            
            # Calculate time spent above/below each threshold
            time_result, percentage_result = calculate_time_threshold(df, col, temp, is_heat_analysis)
            
            if is_heat_analysis:
                stats_results[col][f'Time above {temp}°C'] = time_result
                stats_results[col][f'Percentage above {temp}°C'] = percentage_result
            else:
                stats_results[col][f'Time below {temp}°C'] = time_result
                stats_results[col][f'Percentage below {temp}°C'] = percentage_result
    
    return stats_results


def plot_temperature_trends(df, temp_columns):
    """
    Create a line plot of temperature trends.
    
    Args:
        df (DataFrame): Processed temperature data
        temp_columns (list): List of temperature columns to plot
        
    Returns:
        matplotlib.figure.Figure: Figure object with the plot
    """
    # Check if DataFrame is empty
    if df.empty:
        # Return empty figure if no data
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No data to plot', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        return fig
    
    # Limit figure size to prevent oversized plots
    figsize = (min(14, len(df) * 0.01), 8)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create x-axis for time (seconds)
    x = np.arange(len(df))
    
    # Determine appropriate marker frequency to avoid overcrowding
    marker_every = max(1, len(x) // 100)
    
    # Color cycle for better differentiation with many channels
    colors = plt.cm.tab20(np.linspace(0, 1, len(temp_columns)))
    
    # Plot each temperature column that exists
    for i, col in enumerate(temp_columns):
        if col in df.columns and not df[col].isna().all():
            try:
                # Use markevery to reduce the number of markers for large datasets
                ax.plot(x, df[col], label=col, markevery=marker_every, 
                        color=colors[i % len(colors)])
            except Exception as e:
                st.warning(f"Could not plot {col}: {str(e)}")
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Trends')
    
    # Improve legend layout for many columns
    if len(temp_columns) > 8:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                  ncol=min(8, len(temp_columns)), fontsize='small')
    else:
        ax.legend()
    
    ax.grid(True)
    
    return fig

def display_data_tab(df, temp_columns):
    """
    Display data in the Data tab.
    
    Args:
        df (DataFrame): Processed temperature data
        temp_columns (list): List of temperature columns
    """
    st.subheader("Raw Data")
    
    # Display shape information
    st.write(f"Data shape: {df.shape} (rows, columns)")
    
    # Show data types
    st.subheader("Data Types")
    st.write(df.dtypes)
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())
    
    # Show the actual data
    st.subheader("Data Preview")
    st.dataframe(df.head(100))
    
    # Allow downloading the full data
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Full Data as CSV",
        data=csv,
        file_name='temperature_data.csv',
        mime='text/csv',
    )

def display_statistics_tab(df, stats, temp_columns, is_heat_analysis=True):
    """
    Display statistics in the Statistics tab.
    
    Args:
        df (DataFrame): Processed temperature data
        stats (dict): Statistics calculated for each channel
        temp_columns (list): List of temperature columns
        is_heat_analysis (bool): Whether to display heat or cold analysis metrics
    """
    analysis_type = "Heat" if is_heat_analysis else "Cold"
    st.subheader(f"Temperature Statistics ({analysis_type} Analysis)")
    
    # If there are many columns, we may need to paginate the statistics
    # or allow selecting a subset of columns to view
    if len(temp_columns) > 8:
        st.info(f"You have {len(temp_columns)} temperature channels. "
                f"Consider selecting fewer channels for clearer analysis.")
        
        # Option to filter columns for statistics view
        show_all_stats = st.checkbox("Show statistics for all columns", value=False)
        
        if not show_all_stats:
            # Allow selecting a subset of columns
            selected_for_stats = st.multiselect(
                "Select columns to show statistics for:",
                options=temp_columns,
                default=temp_columns[:min(8, len(temp_columns))]
            )
            
            if selected_for_stats:
                # Filter stats to only selected columns
                filtered_stats = {col: stats[col] for col in selected_for_stats if col in stats}
                stats_to_display = filtered_stats
            else:
                stats_to_display = stats
        else:
            stats_to_display = stats
    else:
        stats_to_display = stats
    
    # Convert stats to DataFrame for display
    stats_df = pd.DataFrame({key: {k: v for k, v in values.items()} 
                          for key, values in stats_to_display.items()}).T
    
    st.dataframe(stats_df)
    
    # Create downloadable report
    report = io.StringIO()
    report.write(f"Temperature Analysis Report - {analysis_type} Analysis\n\n")
    
    # Convert all stats (not just filtered ones) to CSV for the report
    pd.DataFrame({key: {k: v for k, v in values.items()} 
               for key, values in stats.items()}).T.to_csv(report)
    
    st.download_button(
        label=f"Download {analysis_type} Statistics Report",
        data=report.getvalue(),
        file_name=f'temperature_statistics_{analysis_type.lower()}.csv',
        mime='text/csv',
    )
    
    # Get thresholds
    threshold_list = st.session_state.get('threshold_list', [38,41,43])
    
    # Add time-to-temperature visualization
    if is_heat_analysis:
        st.subheader("Time to Reach Temperature Thresholds")
        time_key_pattern = "Time to reach {temp}°C"
    else:
        st.subheader("Time to Reach Below Temperature Thresholds")
        time_key_pattern = "Time to reach below {temp}°C"
    
    # Create a table with columns for each channel and rows for thresholds
    time_data = {}
    for temp in threshold_list:
        time_data[f"{temp}°C"] = {}
        key = time_key_pattern.format(temp=temp)
        for col in stats_to_display:
            if col in stats:
                if key in stats[col]:
                    time_data[f"{temp}°C"][col] = stats[col][key]
                else:
                    time_data[f"{temp}°C"][col] = "N/A"
            else:
                time_data[f"{temp}°C"][col] = "N/A"
    
    time_df = pd.DataFrame(time_data).T
    st.table(time_df)
    
    # Add time-above/below-temperature visualization
    if is_heat_analysis:
        st.subheader("Time Spent Above Temperature Thresholds")
        time_above_key = "Time above {temp}°C"
    else:
        st.subheader("Time Spent Below Temperature Thresholds")
        time_above_key = "Time below {temp}°C"
    
    # Create a table with columns for each channel and rows for thresholds
    time_above_data = {}
    for temp in threshold_list:
        time_above_data[f"{temp}°C"] = {}
        key = time_above_key.format(temp=temp)
        for col in stats_to_display:
            if col in stats:
                if key in stats[col]:
                    time_above_data[f"{temp}°C"][col] = stats[col][key]
                else:
                    time_above_data[f"{temp}°C"][col] = "N/A"
            else:
                time_above_data[f"{temp}°C"][col] = "N/A"
    
    time_above_df = pd.DataFrame(time_above_data).T
    st.table(time_above_df)
    
    # Add percentage-above/below-temperature visualization
    if is_heat_analysis:
        st.subheader("Percentage of Time Above Temperature Thresholds")
        percentage_key = "Percentage above {temp}°C"
    else:
        st.subheader("Percentage of Time Below Temperature Thresholds")
        percentage_key = "Percentage below {temp}°C"
    
    # Create a table with columns for each channel and rows for thresholds
    percentage_above_data = {}
    for temp in threshold_list:
        percentage_above_data[f"{temp}°C"] = {}
        key = percentage_key.format(temp=temp)
        for col in stats_to_display:
            if col in stats:
                if key in stats[col]:
                    percentage_above_data[f"{temp}°C"][col] = stats[col][key]
                else:
                    percentage_above_data[f"{temp}°C"][col] = "N/A"
            else:
                percentage_above_data[f"{temp}°C"][col] = "N/A"
    
    percentage_above_df = pd.DataFrame(percentage_above_data).T
    st.table(percentage_above_df)


def display_visualization_tab(df, stats, temp_columns, is_heat_analysis=True):
    """Display visualizations in the Visualization tab."""
    analysis_type = "Heat" if is_heat_analysis else "Cold"
    st.subheader(f"Temperature Visualization ({analysis_type} Analysis)")
    
    # Set a reasonable maximum figure size and control DPI
    max_figsize = 14
    figure_dpi = 100
    
    # Handle many columns for visualization
    if len(temp_columns) > 8:
        st.info(f"You have {len(temp_columns)} temperature channels. For clearer visualizations, "
                f"consider selecting a subset of channels to display.")
        
        # Allow selecting a subset of columns for visualization
        show_all_viz = st.checkbox("Show all channels in visualizations", value=False)
        
        if not show_all_viz:
            # Get available temperature columns
            available_temp_cols = [col for col in temp_columns 
                                if col in df.columns and not df[col].isna().all()]
            
            # Allow selecting a subset of columns for visualization
            selected_for_viz = st.multiselect(
                "Select columns to visualize:",
                options=available_temp_cols,
                default=available_temp_cols[:min(8, len(available_temp_cols))]
            )
            
            if selected_for_viz:
                viz_columns = selected_for_viz
            else:
                viz_columns = temp_columns
        else:
            viz_columns = temp_columns
    else:
        viz_columns = temp_columns
    
    # Limit the number of data points to prevent oversized plots
    max_data_points = 10000
    if len(df) > max_data_points:
        st.warning(f"Dataset contains {len(df)} points, which may be too large for visualization. Downsampling to {max_data_points} points.")
        # Use a simple downsampling by taking every nth row
        sampling_rate = len(df) // max_data_points + 1
        df_plot = df.iloc[::sampling_rate].copy()
    else:
        df_plot = df.copy()
    
    # Plot temperature trends with controlled figure size and DPI
    try:
        # Modified function call to plot_temperature_trends with selected columns
        fig = plot_temperature_trends(df_plot, viz_columns)
        st.pyplot(fig, dpi=figure_dpi)
    except Exception as e:
        st.error(f"Error creating temperature trends plot: {str(e)}")
    
    # Additional visualizations
    st.subheader("Temperature Distribution")
    
    # Get available temperature columns
    available_temp_cols = [col for col in viz_columns 
                        if col in df.columns and not df[col].isna().all()]
    
    if not available_temp_cols:
        st.warning("No temperature columns available for histogram visualization.")
    else:
        try:
            # Limit the number of columns to display to prevent oversized figures
            max_cols_to_display = 14
            if len(available_temp_cols) > max_cols_to_display:
                st.warning(f"Limiting histogram display to {max_cols_to_display} columns.")
                display_cols = available_temp_cols[:max_cols_to_display]
            else:
                display_cols = available_temp_cols
                
            # Create a subplot grid for histograms with controlled size
            # Calculate best grid layout based on number of columns
            if len(display_cols) <= 16:
                n_rows, n_cols = 1, len(display_cols)
            else:
                n_rows = (len(display_cols) + 3) // 4  # Ceiling division
                n_cols = min(4, len(display_cols))
            
            fig_width = min(4 * n_cols, max_figsize)
            fig_height = min(3 * n_rows, max_figsize)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=figure_dpi)
            
            # Handle case with only one column
            if len(display_cols) == 1:
                axes = np.array([axes])
            
            # Flatten the axes array for easier indexing
            axes = np.array(axes).flatten()
            
            for i, col in enumerate(display_cols):
                try:
                    axes[i].hist(df_plot[col], bins=20, alpha=0.7)
                    axes[i].set_title(f'{col}', fontsize=9)
                    axes[i].set_xlabel('Temperature (°C)', fontsize=8)
                    axes[i].set_ylabel('Frequency', fontsize=8)
                    axes[i].tick_params(axis='both', which='major', labelsize=7)
                    
                    # Add reference lines if stats are available
                    if col in stats:
                        if 'Average' in stats[col]:
                            axes[i].axvline(stats[col]['Average'], color='r', linestyle='--', label='Mean')
                        if 'Mode' in stats[col]:
                            axes[i].axvline(stats[col]['Mode'], color='g', linestyle='--', label='Mode')
                        axes[i].legend(fontsize=7)
                except Exception as e:
                    st.error(f"Error creating histogram for {col}: {str(e)}")
            
            # Hide unused subplots
            for i in range(len(display_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig, dpi=figure_dpi)
        except Exception as e:
            st.error(f"Error creating histogram plots: {str(e)}")
    
    # Add temperature vs time visualization with controlled size
    st.subheader("Temperature vs Time")
    
    try:
        # Create a line plot showing temperature over time with limited size
        fig, ax = plt.subplots(figsize=(min(14, max_figsize), min(8, max_figsize)), dpi=figure_dpi)
        
        # Create x-axis for time (seconds)
        x = np.arange(len(df_plot))
        
        # Color cycle for many channels
        colors = plt.cm.tab20(np.linspace(0, 1, len(available_temp_cols)))
        
        for i, col in enumerate(available_temp_cols):
            try:
                ax.plot(x, df_plot[col], marker='.', linestyle='-', 
                        label=col, markevery=max(1, len(x)//100),
                        color=colors[i % len(colors)])
            except Exception as e:
                st.warning(f"Could not plot {col}: {str(e)}")
        
        # Add threshold lines
        threshold_list = st.session_state.get('threshold_list', [38,41,43])
        for t in threshold_list:
            ax.axhline(y=t, color='gray', linestyle='--', alpha=0.5)
            ax.text(x.max(), t, f"{t}°C", va='center', ha='left', fontsize=8)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature Change Over Time')
        
        # Improve legend layout for many columns
        if len(available_temp_cols) > 8:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                      ncol=min(8, len(available_temp_cols)), fontsize='small')
        else:
            ax.legend()
            
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig, dpi=figure_dpi)
    except Exception as e:
        st.error(f"Error creating temperature vs time plot: {str(e)}")
    
    # Add rate of temperature change visualization with controlled size
    st.subheader("Rate of Temperature Change")
    
    try:
        fig, ax = plt.subplots(figsize=(min(14, max_figsize), min(8, max_figsize)), dpi=figure_dpi)
        
        # Color cycle for many channels
        colors = plt.cm.tab20(np.linspace(0, 1, len(available_temp_cols)))
        
        for i, col in enumerate(available_temp_cols):
            try:
                # Calculate temperature change rate (°C per second)
                temp_values = df_plot[col]
                
                # Calculate temperature change rate using rolling window
                temp_diff = temp_values.diff()
                
                # Plot the rate of change
                valid_mask = ~temp_diff.isna()
                ax.plot(x[valid_mask], temp_diff[valid_mask], marker='.', 
                        label=f"{col} Rate", markevery=max(1, len(x)//100),
                        color=colors[i % len(colors)])
            except Exception as e:
                st.warning(f"Could not calculate rate for {col}: {str(e)}")
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Rate of Change (°C/second)')
        ax.set_title('Temperature Change Rate')
        
        # Improve legend layout for many columns
        if len(available_temp_cols) > 8:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                      ncol=min(8, len(available_temp_cols)), fontsize='small')
        else:
            ax.legend()
            
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig, dpi=figure_dpi)
    except Exception as e:
        st.error(f"Error creating rate of change plot: {str(e)}")
    
    # Add time-above/below-threshold visualization
    if is_heat_analysis:
        st.subheader("Time Spent Above Temperature Thresholds")
        time_key = "Time above {temp}°C"
    else:
        st.subheader("Time Spent Below Temperature Thresholds")
        time_key = "Time below {temp}°C"
    
    # Get thresholds
    threshold_list = st.session_state.get('threshold_list', [29, 30, 31, 32, 33, 34])
    
    try:
        # Create a bar chart showing time spent above/below each threshold for each channel
        fig, ax = plt.subplots(figsize=(min(14, max_figsize), min(8, max_figsize)), dpi=figure_dpi)
        
        # Number of channels and thresholds
        num_channels = len(available_temp_cols)
        num_thresholds = len(threshold_list)
        
        # Set up bar positions
        bar_width = 0.8 / num_channels
        positions = np.arange(num_thresholds)
        
        # Color cycle for many channels
        colors = plt.cm.tab20(np.linspace(0, 1, num_channels))
        
        for i, col in enumerate(available_temp_cols):
            if col in stats:
                # Extract time spent above/below each threshold (in seconds)
                times_above = []
                for temp in threshold_list:
                    key = time_key.format(temp=temp)
                    if key in stats[col]:
                        # Try to extract samples from format like "m:ss (n samples)"
                        time_str = stats[col][key]
                        try:
                            samples = int(time_str.split('(')[1].split(' ')[0])
                        except:
                            samples = 0
                        times_above.append(samples)
                    else:
                        times_above.append(0)
                
                # Plot the bars
                bar_pos = positions + (i - num_channels/2 + 0.5) * bar_width
                ax.bar(bar_pos, times_above, width=bar_width, label=col, 
                       color=colors[i % len(colors)])
        
        # Set labels and title
        ax.set_xlabel('Temperature Threshold (°C)')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Time Spent {"Above" if is_heat_analysis else "Below"} Temperature Thresholds')
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{temp}°C" for temp in threshold_list])
        
        # Improve legend layout for many columns
        if len(available_temp_cols) > 8:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                      ncol=min(8, len(available_temp_cols)), fontsize='small')
        else:
            ax.legend()
            
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig, dpi=figure_dpi)
    except Exception as e:
        st.error(f"Error creating threshold bar chart: {str(e)}")

def main():
    """
    Main function to run the Streamlit app with support for multiple files and file combination.
    """
    st.title("Temperature Analysis Tool")
    
    # Add analysis type selection
    st.sidebar.header("Analysis Type")
    is_heat_analysis = st.sidebar.checkbox("Heat Analysis", value=True, help="When checked, analyzes time above thresholds. When unchecked, analyzes time below thresholds.")
    
    analysis_type = "Heat" if is_heat_analysis else "Cold"
    st.sidebar.write(f"Current Mode: **{analysis_type} Analysis**")
    
    if is_heat_analysis:
        st.sidebar.info("Heat Analysis: Measuring time **above** temperature thresholds and time to **reach** thresholds.")
    else:
        st.sidebar.info("Cold Analysis: Measuring time **below** temperature thresholds and time to **reach below** thresholds.")
    
    # Add custom thresholds input with different defaults based on analysis type
    default_value = "38,41,43" if is_heat_analysis else "8,10,12"
    
    custom_thresholds = st.sidebar.text_input(
        "Custom temperature thresholds (comma-separated, e.g., 29,31,33)", 
        value=default_value
    )
    try:
        # Parse custom thresholds
        threshold_list = [float(t.strip()) for t in custom_thresholds.split(",") if t.strip()]
        st.session_state['threshold_list'] = threshold_list
    except:
        st.sidebar.error("Invalid threshold format. Using default thresholds.")
        threshold_list = [38, 41, 43] if is_heat_analysis else [8,10,12]
        st.session_state['threshold_list'] = threshold_list
    
    # File upload section
    st.sidebar.header("Upload Data")
    uploaded_files = st.sidebar.file_uploader("Choose temperature data file(s)", type=['txt', 'csv'], accept_multiple_files=True)
    
    # Option to specify delimiter for CSV
    delimiters = st.sidebar.radio(
        "Choose file delimiter (if auto-detection fails):",
        options=[("Auto-detect", None), ("Comma (,)", ","), ("Tab (\\t)", "\t"), ("Space", " ")],
        format_func=lambda x: x[0]
    )
    
    # Add options to specify row indices
    st.sidebar.header("Row Selection")
    header_row_index = st.sidebar.number_input("Header row index (0-based)", min_value=0, value=3)
    data_start_row_index = st.sidebar.number_input("Data start row index (0-based)", min_value=0, value=5)
    
    # Process data if files are uploaded
    if not uploaded_files:
        st.info("Please upload one or more data files")
        return
        
    # Create a dictionary to store processed data for each file
    processed_files = {}
    
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        try:
            # Read the file
            data_text = uploaded_file.getvalue().decode('utf-8')
            
            # Create a unique key for each file's column renaming section
            file_key = uploaded_file.name.replace(".", "_").replace(" ", "_")
            
            # Add column renaming section for this file
            with st.sidebar.expander(f"Rename columns for {uploaded_file.name}"):
                st.write("Enter new names for the first 8 columns:")
                
                # Create input fields for column renaming
                column_renames = {}
                
                # Default column names - these are just placeholders until we read the actual file
                column_names = [f"Column_{i+1}" for i in range(8)]
                
                # Create input fields for the first 6 columns
                new_column_names = []
                for i in range(8):
                    if i < len(column_names):
                        col_name = column_names[i]
                        new_name = st.text_input(f"Column {i+1}: {col_name}", 
                                               value="", 
                                               key=f"col_rename_{file_key}_{i}")
                        if new_name.strip():
                            column_renames[col_name] = new_name
                            new_column_names.append(new_name)
                        else:
                            new_column_names.append(col_name)
            
            # First try parsing with pandas directly using specified delimiter
            try:
                st.write(f"Processing file: **{uploaded_file.name}**")
                st.write("Attempting to read file with pandas...")
                buffer = io.StringIO(data_text)
                # Use the selected delimiter, or None for auto-detection
                delimiter = delimiters[1] if delimiters[0] != "Auto-detect" else None
                df = pd.read_csv(buffer, sep=delimiter, engine='python', skiprows=header_row_index)
                
                # Check if the column names from the actual file 
                # are different from our placeholders
                actual_column_names = df.columns.tolist()
                updated_column_renames = {}
                
                # Map user-provided names to actual column names
                for i, new_name in enumerate(new_column_names):
                    if i < len(actual_column_names) and new_name != column_names[i]:
                        updated_column_renames[actual_column_names[i]] = new_name
                
                # Apply column renaming if any columns were renamed
                if updated_column_renames:
                    df = df.rename(columns=updated_column_renames)
                    st.write("Columns renamed successfully!")
                
                st.write(f"Successfully read file with pandas. Shape: {df.shape}")
                st.write("First 5 rows:")
                st.write(df.head())
                experiment_info = uploaded_file.name
            except Exception as e:
                st.warning(f"Direct pandas read failed: {str(e)}")
                st.write("Falling back to manual parsing method...")
                df, experiment_info = process_data(data_text, header_row_index, data_start_row_index, column_renames)
            
            # Store the processed data
            processed_files[uploaded_file.name] = {
                'df': df,
                'experiment_info': experiment_info
            }
            
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
    
    # Check if we have any processed files
    if not processed_files:
        st.error("No files could be processed. Please check your data format.")
        return
    
    # New feature: File combination option
    if len(processed_files) > 1:  # Only show if multiple files are uploaded
        st.sidebar.header("File Combination")
        enable_combination = st.sidebar.checkbox("Enable file combination", value=False)
        
        if enable_combination:
            st.sidebar.subheader("Select files to combine")
            file_names = list(processed_files.keys())
            
            # Allow selecting which files to combine
            selected_files = st.sidebar.multiselect(
                "Select files to combine",
                options=file_names,
                default=file_names
            )
            
            # For each selected file, allow selecting which columns to include
            column_selections = {}
            if selected_files:
                st.sidebar.subheader("Select columns to combine")
                
                for file_name in selected_files:
                    file_data = processed_files[file_name]
                    df = file_data['df']
                    
                    with st.sidebar.expander(f"Columns from {file_name}"):
                        # Find numeric columns that might be temperature data
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        # Filter out columns that look like time/sample
                        numeric_cols = [col for col in numeric_cols if not any(word in str(col).lower() 
                                                                        for word in ['sample', 'time', 'second', 'minute'])]
                        
                        # Select columns to include
                        selected_cols = st.multiselect(
                            f"Select columns from {file_name}",
                            options=df.columns.tolist(),
                            default=numeric_cols[:min(8, len(numeric_cols))],
                            key=f"select_cols_{file_name.replace('.', '_').replace(' ', '_')}"
                        )
                        
                        column_selections[file_name] = selected_cols
                
                # Add a button to trigger the combination
                if st.sidebar.button("Combine Selected Files"):
                    # Combine the files
                    combined_df, combined_info = merge_selected_files(
                        processed_files, 
                        selected_files, 
                        column_selections
                    )
                    
                    if not combined_df.empty:
                        # Add the combined file to processed_files
                        combined_file_name = "Combined_Data"
                        processed_files[combined_file_name] = {
                            'df': combined_df,
                            'experiment_info': combined_info
                        }
                        
                        st.success(f"Created combined file: {combined_file_name}")
                        # Store in session state that we've selected this file
                        st.session_state['selected_file'] = combined_file_name

                    if "Combined_Data" in processed_files:
                        combined_df = processed_files["Combined_Data"]['df']
                        
                        # Get all numeric columns except Sample as potential temperature columns
                        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
                        if 'Sample' in numeric_cols:
                            numeric_cols.remove('Sample')
                        
                        # Store as temperature columns for the combined file
                        if numeric_cols:
                            st.write("Setting temperature columns for combined data:", numeric_cols)
                            st.session_state['combined_temp_columns'] = numeric_cols
                        else:
                            st.error("No numeric columns found in combined data")

    
    # Create a file selector
    st.header("Select File to Analyze")
    file_names = list(processed_files.keys())
    
    # Make sure we have files to work with
    if not file_names:
        st.error("No processed files available.")
        return
    
    # Default to the combined file if it exists and was just created
    default_file = st.session_state.get('selected_file', file_names[0])
    if default_file not in file_names:
        default_file = file_names[0]
    
    # Always define selected_file outside of any conditional blocks
    selected_file = st.selectbox("Choose a file to view analysis", 
                              file_names,
                              index=file_names.index(default_file))
    
    # Update session state with current selection
    st.session_state['selected_file'] = selected_file
    
    # For combined data in cold analysis mode, adjust thresholds if needed
    if selected_file == "Combined_Data" and not is_heat_analysis:
        # Check if we're using the default heat thresholds
        current_thresholds = st.session_state.get('threshold_list', [])
        if all(t >= 35 for t in current_thresholds):  # If using high thresholds in cold mode
            # Set appropriate cold thresholds
            cold_thresholds = [8, 10, 12]
            st.session_state['threshold_list'] = cold_thresholds
            st.sidebar.info("Automatically adjusted thresholds for cold analysis. You can modify these in the sidebar.")
    
    # Check if selected file exists in processed_files
    if selected_file not in processed_files:
        st.error(f"Selected file '{selected_file}' not found in processed files.")
        return
    
    # Get the selected file's data
    selected_data = processed_files[selected_file]
    df = selected_data['df']
    experiment_info = selected_data['experiment_info']
    
    # Check if this is the combined file
    is_combined = selected_file == "Combined_Data"
    
    # Identify temperature columns - get all temperature-related columns
    temp_columns = []
    
    # For combined data, use all numeric columns (except Sample)
    if is_combined and 'combined_temp_columns' in st.session_state:
        temp_columns = st.session_state['combined_temp_columns']
    else:
        # Otherwise, look for typical temperature columns
        for col in df.columns:
            # Look for columns with temperature-related names
            if ('Channel' in col or 'temp' in col.lower() or '°C' in col):
                temp_columns.append(col)
        
        # If no specific temperature columns found, use all numeric columns
        if not temp_columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            # Filter out columns that look like time/sample
            numeric_cols = [col for col in numeric_cols if not any(word in str(col).lower() 
                                                               for word in ['sample', 'time', 'second', 'minute'])]
            temp_columns = list(numeric_cols)
    
    # Manual column selection for visualization
    # Get all numeric columns as options
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out columns that look like time
    numeric_cols = [col for col in numeric_cols if not any(word in str(col).lower() 
                                                        for word in ['sample', 'time', 'second', 'minute'])]
    
    # Default to the auto-detected temperature columns
    default_selections = temp_columns if all(col in numeric_cols for col in temp_columns) else numeric_cols
    
    # Add multiselect for columns
    st.header("Select Columns for Visualization")
    selected_columns = st.multiselect(
        "Choose which columns to include in graphs and analysis:",
        options=numeric_cols,
        default=default_selections
    )
    
    # Update temp_columns with user selection if any are selected
    if selected_columns:
        temp_columns = selected_columns
        # Add this line to save selected columns to session state
        st.session_state['temp_columns'] = temp_columns
    
    st.write(f"Using columns for visualization: {', '.join(temp_columns)}")
    
    # Display basic information
    st.header("Temperature Analysis")
    st.subheader(f"File: {selected_file}")
    st.subheader(f"Data source: {experiment_info}")
    st.write(f"Selected temperature channels: {', '.join(temp_columns)}")
    analysis_mode = "Heat" if is_heat_analysis else "Cold"
    st.write(f"Analysis mode: {analysis_mode}")
    
    # Calculate statistics based on selected analysis type
    stats = calculate_statistics(df, is_heat_analysis)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Data", "Statistics", "Visualization"])
    
    # Display content in each tab
    with tab1:
        display_data_tab(df, temp_columns)
    
    with tab2:
        display_statistics_tab(df, stats, temp_columns, is_heat_analysis)
    
    with tab3:
        display_visualization_tab(df, stats, temp_columns, is_heat_analysis)

# Run the application
if __name__ == "__main__":
    main()
