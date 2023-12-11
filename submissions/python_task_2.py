import pandas as pd
import networkx as nx
from datetime import time

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    # Read the dataset into a DataFrame

    # Create a directed graph from the DataFrame
    G = nx.from_pandas_edgelist(df, 'id_start', 'id_end', 'distance')
    
    # Calculate shortest path lengths considering cumulative distances
    distance_matrix = nx.floyd_warshall_numpy(G, weight='distance')

    dist = pd.DataFrame(distance_matrix, index=G.nodes(), columns=G.nodes())
    return dist


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    # Create a graph from the distance matrix
    G = nx.from_pandas_adjacency(df)

    # Get edges and their weights (distances)
    edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]

    # Create a DataFrame from the edges
    unrolled_df = pd.DataFrame(edges, columns=['id_start', 'id_end', 'distance'])

    data = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']].reset_index(drop=True)

    return data


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    # Filter rows based on the reference_value in the id_start column
    reference_df = df[df['id_start'] == reference_id]

    # Calculate average distance for the reference value
    average_distance = reference_df['distance'].mean()

    # Calculate the threshold range (10% of the average distance)
    threshold = 0.1 * average_distance

    # Find id_start values within the 10% threshold range
    within_threshold = df[(df['id_start'] != reference_id) & 
                                 (df['distance'] >= (average_distance - threshold)) &
                                 (df['distance'] <= (average_distance + threshold))]['id_start'].unique()

    # Sort and return the list of id_start values within the threshold
    data = sorted(within_threshold)
    data = pd.DataFrame(data, columns=['id_start'])
    return data


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    # Rate coefficients for different vehicle types
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type based on the distance
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    start_day_list, start_time_list, end_day_list, end_time_list = [], [], [], []

    time_range = [time(0, 0), time(10, 0), time(18, 0), time(23, 59, 59)]

    for pair, group in df.groupby(['id_start', 'id_end']):
        id_start, id_end = pair

        for i in range(len(time_range) - 1):
            start_time = time_range[i]
            end_time = time_range[i + 1]

            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                start_day_list.append(day)
                start_time_list.append(start_time)
                end_day_list.append(day)
                end_time_list.append(end_time)

    # Truncate lists to match DataFrame length
    num_rows = len(df)
    start_day_list = start_day_list[:num_rows]
    start_time_list = start_time_list[:num_rows]
    end_day_list = end_day_list[:num_rows]
    end_time_list = end_time_list[:num_rows]
    df['start_day'] = start_day_list
    df['start_time'] = start_time_list
    df['end_day'] = end_day_list
    df['end_time'] = end_time_list

    # Define function to calculate toll rates based on time intervals
    def calculate_toll_rate(row):
        weekday_discount = {0.8: [(time(0, 0), time(10, 0)), (time(18, 0), time(23, 59, 59))],
                            1.2: [(time(10, 0), time(18, 0))]}
        weekend_discount = 0.7

        start_time = row['start_time']
        end_time = row['end_time']
        start_day = row['start_day']
        end_day = row['end_day']

        if start_day in ['Saturday', 'Sunday'] or end_day in ['Saturday', 'Sunday']:
            return weekend_discount
        else:
            for discount, intervals in weekday_discount.items():
                for interval in intervals:
                    if interval[0] <= start_time <= interval[1] or interval[0] <= end_time <= interval[1]:
                        return discount
            return 1.0  # Default factor if not within specified intervals

    # Apply the toll rate calculation function to create new columns for vehicle types
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        for i in range(len(time_range) - 1):
            start_time = time_range[i]
            end_time = time_range[i + 1]

            mask = ((df['start_day'] == day) & (df['start_time'] == start_time) &
                    (df['end_day'] == day) & (df['end_time'] == end_time))

            discount = df[mask].apply(calculate_toll_rate, axis=1).values[0]

            for col in ['moto', 'car', 'rv', 'bus', 'truck']:
                df.loc[mask, col] *= discount

    return df
