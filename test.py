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
                
                if sample_col is not None and pd.api.types.is_numeric_dtype(df[sample_col]):
                    # Use the actual sample number from the data
                    time_seconds = float(df.iloc[first_index][sample_col])
                else:
                    # Time in seconds is just the row index (assuming 1 Hz sampling)
                    time_seconds = first_index
                
                # Format as time (MM:SS)
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
        
    # Get temperature column names - use first 4 channels
    temp_columns = []
    channel_count = 0
    
    # First, try to find columns with expected temperature names
    for col in df.columns:
        col_str = str(col).lower()
        # Look for columns with temperature-related names
        if ('channel' in col_str and ('1' in col_str or '2' in col_str or '3' in col_str or '4' in col_str)) or \
           ('temp' in col_str) or ('°c' in col_str) or ('temperature' in col_str):
            # Make sure it's numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                temp_columns.append(col)
                channel_count += 1
                if channel_count >= 4:  # Limit to 4 channels
                    break
    
    # If we couldn't find 4 columns that way, use the first 4 numeric columns
    if channel_count < 4:
        st.warning(f"Only found {channel_count} specific temperature columns. Adding more from numeric columns.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in temp_columns and channel_count < 4:
                # Skip columns that look like sample numbers or time
                col_str = str(col).lower()
                if 'sample' in col_str or 'time' in col_str or 'second' in col_str or 'minute' in col_str:
                    continue
                temp_columns.append(col)
                channel_count += 1
                if channel_count >= 4:
                    break
    
    st.write(f"Using channels: {temp_columns}")
    
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
        thresholds = st.session_state.get('threshold_list', [29, 30, 31, 32, 33, 34])
        
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
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No data to plot', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create x-axis for time (seconds)
    x = np.arange(len(df))
    
    # Plot each temperature column that exists
    for col in temp_columns:
        if col in df.columns and not df[col].isna().all():
            try:
                ax.plot(x, df[col], label=col)
            except Exception as e:
                st.warning(f"Could not plot {col}: {str(e)}")
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Trends')
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
    st.subheader("Temperature Data")
    
    # Display only the relevant columns
    if not temp_columns:
        st.dataframe(df)
    else:
        st.dataframe(df[temp_columns])
    
    # Allow download of processed data
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download processed data as CSV",
        data=csv,
        file_name='processed_temperature_data.csv',
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
    
    # Convert stats to DataFrame for display
    stats_df = pd.DataFrame({key: {k: v for k, v in values.items()} 
                          for key, values in stats.items()}).T
    
    st.dataframe(stats_df)
    
    # Create downloadable report
    report = io.StringIO()
    report.write(f"Temperature Analysis Report - {analysis_type} Analysis\n\n")
    
    stats_df.to_csv(report)
    
    st.download_button(
        label=f"Download {analysis_type} Statistics Report",
        data=report.getvalue(),
        file_name=f'temperature_statistics_{analysis_type.lower()}.csv',
        mime='text/csv',
    )
    
    # Get thresholds
    threshold_list = st.session_state.get('threshold_list', [29, 30, 31, 32, 33, 34])
    
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
        for col in temp_columns:
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
        for col in temp_columns:
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
        for col in temp_columns:
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
    """
    Display visualizations in the Visualization tab.
    
    Args:
        df (DataFrame): Processed temperature data
        stats (dict): Statistics calculated for each channel
        temp_columns (list): List of temperature columns
        is_heat_analysis (bool): Whether to display heat or cold analysis metrics
    """
    analysis_type = "Heat" if is_heat_analysis else "Cold"
    st.subheader(f"Temperature Visualization ({analysis_type} Analysis)")
    
    # Plot temperature trends
    fig = plot_temperature_trends(df, temp_columns)
    st.pyplot(fig)
    
    # Additional visualizations
    st.subheader("Temperature Distribution")
    
    # Get available temperature columns
    available_temp_cols = [col for col in temp_columns 
                        if col in df.columns and not df[col].isna().all()]
    
    if not available_temp_cols:
        st.warning("No temperature columns available for histogram visualization.")
    else:
        # Create a subplot for histograms of each temperature column
        fig, axes = plt.subplots(1, len(available_temp_cols), figsize=(5 * len(available_temp_cols), 5))
        
        # Handle case with only one column
        if len(available_temp_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(available_temp_cols):
            try:
                axes[i].hist(df[col], bins=20, alpha=0.7)
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_xlabel('Temperature (°C)')
                axes[i].set_ylabel('Frequency')
                
                # Add reference lines if stats are available
                if col in stats:
                    if 'Average' in stats[col]:
                        axes[i].axvline(stats[col]['Average'], color='r', linestyle='--', label='Mean')
                    if 'Mode' in stats[col]:
                        axes[i].axvline(stats[col]['Mode'], color='g', linestyle='--', label='Mode')
                    axes[i].legend()
            except Exception as e:
                st.error(f"Error creating histogram for {col}: {str(e)}")
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Add heating/cooling rate visualization
    st.subheader("Temperature vs Time")
    
    # Create a line plot showing temperature over time
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create x-axis for time (seconds)
    x = np.arange(len(df))
    
    for col in available_temp_cols:
        try:
            ax.plot(x, df[col], marker='.', linestyle='-', label=col)
        except Exception as e:
            st.warning(f"Could not plot {col}: {str(e)}")
    
    # Add threshold lines
    threshold_list = st.session_state.get('threshold_list', [29, 30, 31, 32, 33, 34])
    for t in threshold_list:
        ax.axhline(y=t, color='gray', linestyle='--', alpha=0.5)
        ax.text(x.max(), t, f"{t}°C", va='center', ha='left', fontsize=8)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Change Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add rate of temperature change visualization
    st.subheader("Rate of Temperature Change")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for col in available_temp_cols:
        try:
            # Calculate temperature change rate (°C per second)
            temp_values = df[col]
            
            # Calculate temperature change rate using rolling window
            temp_diff = temp_values.diff()
            
            # Plot the rate of change
            valid_mask = ~temp_diff.isna()
            ax.plot(x[valid_mask], temp_diff[valid_mask], marker='.', label=f"{col} Rate")
        except Exception as e:
            st.warning(f"Could not calculate rate for {col}: {str(e)}")
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Rate of Change (°C/second)')
    ax.set_title('Temperature Change Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add time-above/below-threshold visualization
    if is_heat_analysis:
        st.subheader("Time Spent Above Temperature Thresholds")
        time_key = "Time above {temp}°C"
    else:
        st.subheader("Time Spent Below Temperature Thresholds")
        time_key = "Time below {temp}°C"
    
    # Get thresholds
    threshold_list = st.session_state.get('threshold_list', [29, 30, 31, 32, 33, 34])
    
    # Create a bar chart showing time spent above/below each threshold for each channel
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Number of channels and thresholds
    num_channels = len(available_temp_cols)
    num_thresholds = len(threshold_list)
    
    # Set up bar positions
    bar_width = 0.8 / num_channels
    positions = np.arange(num_thresholds)
    
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
            ax.bar(bar_pos, times_above, width=bar_width, label=col)
    
    # Set labels and title
    ax.set_xlabel('Temperature Threshold (°C)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'Time Spent {"Above" if is_heat_analysis else "Below"} Temperature Thresholds')
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{temp}°C" for temp in threshold_list])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)

    import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from scipy import stats

# Set page configuration
st.set_page_config(page_title="Temperature Analysis Tool", layout="wide")

# Rest of your functions, including process_data, find_time_to_reach, calculate_time_threshold, etc.
# ...
# (Keep all your existing code here)
# ...

# Complete the display_visualization_tab function that was cut off
def display_visualization_tab(df, stats, temp_columns, is_heat_analysis=True):
    """
    Display visualizations in the Visualization tab.
    
    Args:
        df (DataFrame): Processed temperature data
        stats (dict): Statistics calculated for each channel
        temp_columns (list): List of temperature columns
        is_heat_analysis (bool): Whether to display heat or cold analysis metrics
    """
    analysis_type = "Heat" if is_heat_analysis else "Cold"
    st.subheader(f"Temperature Visualization ({analysis_type} Analysis)")
    
    # Plot temperature trends
    fig = plot_temperature_trends(df, temp_columns)
    st.pyplot(fig)
    
    # Additional visualizations
    st.subheader("Temperature Distribution")
    
    # Get available temperature columns
    available_temp_cols = [col for col in temp_columns 
                        if col in df.columns and not df[col].isna().all()]
    
    if not available_temp_cols:
        st.warning("No temperature columns available for histogram visualization.")
    else:
        # Create a subplot for histograms of each temperature column
        fig, axes = plt.subplots(1, len(available_temp_cols), figsize=(5 * len(available_temp_cols), 5))
        
        # Handle case with only one column
        if len(available_temp_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(available_temp_cols):
            try:
                axes[i].hist(df[col], bins=20, alpha=0.7)
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_xlabel('Temperature (°C)')
                axes[i].set_ylabel('Frequency')
                
                # Add reference lines if stats are available
                if col in stats:
                    if 'Average' in stats[col]:
                        axes[i].axvline(stats[col]['Average'], color='r', linestyle='--', label='Mean')
                    if 'Mode' in stats[col]:
                        axes[i].axvline(stats[col]['Mode'], color='g', linestyle='--', label='Mode')
                    axes[i].legend()
            except Exception as e:
                st.error(f"Error creating histogram for {col}: {str(e)}")
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Add heating/cooling rate visualization
    st.subheader("Temperature vs Time")
    
    # Create a line plot showing temperature over time
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create x-axis for time (seconds)
    x = np.arange(len(df))
    
    for col in available_temp_cols:
        try:
            ax.plot(x, df[col], marker='.', linestyle='-', label=col)
        except Exception as e:
            st.warning(f"Could not plot {col}: {str(e)}")
    
    # Add threshold lines
    threshold_list = st.session_state.get('threshold_list', [29, 30, 31, 32, 33, 34])
    for t in threshold_list:
        ax.axhline(y=t, color='gray', linestyle='--', alpha=0.5)
        ax.text(x.max(), t, f"{t}°C", va='center', ha='left', fontsize=8)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Change Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add rate of temperature change visualization
    st.subheader("Rate of Temperature Change")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for col in available_temp_cols:
        try:
            # Calculate temperature change rate (°C per second)
            temp_values = df[col]
            
            # Calculate temperature change rate using rolling window
            temp_diff = temp_values.diff()
            
            # Plot the rate of change
            valid_mask = ~temp_diff.isna()
            ax.plot(x[valid_mask], temp_diff[valid_mask], marker='.', label=f"{col} Rate")
        except Exception as e:
            st.warning(f"Could not calculate rate for {col}: {str(e)}")
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Rate of Change (°C/second)')
    ax.set_title('Temperature Change Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add time-above/below-threshold visualization
    if is_heat_analysis:
        st.subheader("Time Spent Above Temperature Thresholds")
        time_key = "Time above {temp}°C"
    else:
        st.subheader("Time Spent Below Temperature Thresholds")
        time_key = "Time below {temp}°C"
    
    # Get thresholds
    threshold_list = st.session_state.get('threshold_list', [29, 30, 31, 32, 33, 34])
    
    # Create a bar chart showing time spent above/below each threshold for each channel
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Number of channels and thresholds
    num_channels = len(available_temp_cols)
    num_thresholds = len(threshold_list)
    
    # Set up bar positions
    bar_width = 0.8 / num_channels
    positions = np.arange(num_thresholds)
    
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
            ax.bar(bar_pos, times_above, width=bar_width, label=col)
    
    # Set labels and title
    ax.set_xlabel('Temperature Threshold (°C)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'Time Spent {"Above" if is_heat_analysis else "Below"} Temperature Thresholds')
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{temp}°C" for temp in threshold_list])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)

# Add the main function - this is what was missing!
# Modify the main function to handle multiple files

def main():
    """
    Main function to run the Streamlit app with support for multiple files.
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
    
    st.sidebar.header("Upload Data")
    # Modified to accept multiple files
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
    
    # Add column renaming section
    st.sidebar.header("Column Renaming")
    st.sidebar.write("Enter new names for the first 6 columns:")
    
    # Create input fields for column renaming
    column_renames = {}
    column_names = []
    
    # Get generic column names for renaming interface
    # Default column names
    column_names = [f"Column_{i+1}" for i in range(6)]
    
    # Create input fields for the first 6 columns
    new_column_names = []
    for i in range(6):
        if i < len(column_names):
            col_name = column_names[i]
            new_name = st.sidebar.text_input(f"Column {i+1}: {col_name}", 
                                           value="", 
                                           key=f"col_rename_{i}")
            if new_name.strip():
                column_renames[col_name] = new_name
                new_column_names.append(new_name)
            else:
                new_column_names.append(col_name)
    
    # Add custom thresholds input
    custom_thresholds = st.sidebar.text_input(
        "Custom temperature thresholds (comma-separated, e.g., 29,31,33)", 
        value="38,41,43"
    )
    try:
        # Parse custom thresholds
        threshold_list = [float(t.strip()) for t in custom_thresholds.split(",") if t.strip()]
        st.session_state['threshold_list'] = threshold_list
    except:
        st.sidebar.error("Invalid threshold format. Using default thresholds.")
        threshold_list = [29, 30, 31, 32, 33, 34]
        st.session_state['threshold_list'] = threshold_list
    
    # Process data if files are uploaded
    if uploaded_files:
        # Create a dictionary to store processed data for each file
        processed_files = {}
        
        # Process each uploaded file
        for uploaded_file in uploaded_files:
            try:
                # Read the file
                data_text = uploaded_file.getvalue().decode('utf-8')
                
                # First try parsing with pandas directly using specified delimiter
                try:
                    st.write(f"Processing file: **{uploaded_file.name}**")
                    st.write("Attempting to read file with pandas...")
                    buffer = io.StringIO(data_text)
                    df = pd.read_csv(buffer, sep=delimiters[1], engine='python', skiprows=header_row_index)
                    
                    # Apply column renaming if any columns were renamed
                    if column_renames:
                        df = df.rename(columns=column_renames)
                        st.write("Columns renamed successfully!")
                    
                    st.write(f"Successfully read file with pandas. Shape: {df.shape}")
                    st.write("First 5 rows:")
                    st.write(df.head())
                    experiment_info = uploaded_file.name
                except Exception as e:
                    st.warning(f"Direct pandas read failed: {str(e)}")
                    st.write("Falling back to custom parsing method...")
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
        
        # Create a file selector
        st.header("Select File to Analyze")
        file_names = list(processed_files.keys())
        selected_file = st.selectbox("Choose a file to view analysis", file_names)
        
        # Get the selected file's data
        selected_data = processed_files[selected_file]
        df = selected_data['df']
        experiment_info = selected_data['experiment_info']
        
        # Identify temperature columns - first 4 channels
        temp_columns = []
        channel_count = 0
        
        for col in df.columns:
            # Look for columns with 'Channel' in the name or temperature column names
            if ('Channel' in col or 'temp' in col.lower() or '°C' in col) and channel_count < 4:
                temp_columns.append(col)
                channel_count += 1
        
        # If no specific temperature columns found, use first 4 numeric columns
        if not temp_columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            temp_columns = list(numeric_cols)[:4]
        
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
    else:
        st.info("Please upload one or more data files")
        return
    
if __name__ == "__main__":
    main()