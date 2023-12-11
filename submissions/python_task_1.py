import pandas as pd


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here

    # Pivot the DataFrame to follow task
    pivoted = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Replace diagonal values with 0
    for i in range(len(pivoted)):
        pivoted.iat[i, i] = 0

    return pivoted


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    df['car_type'] = pd.cut(
        df['car'],
        bins=[float('-inf'), 15, 25, float('inf')],
        labels=['low', 'medium', 'high'],
        right=False
    )

    # Calculate the count of occurrences for each 'car_type' category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    type_counts_sorted = dict(sorted(type_counts.items()))

    return type_counts_sorted


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    bus_mean = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean value
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Calculate average values of 'truck' column for each 'route'
    avg_truck_by_route = df.groupby('route')['truck'].mean()
    # Filter routes where the average of 'truck' column is greater than 7
    filtered_routes = avg_truck_by_route[avg_truck_by_route > 7].index.tolist()

    # Sort the list of routes in ascending order
    filtered_routes.sort()

    return filtered_routes


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    modified_df = matrix.applymap(lambda x: round(x * 0.75, 1) if x > 20 else round(x * 1.25, 1))

    return modified_df


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Convert day names to integers (Monday=0, Tuesday=1, ..., Sunday=6)
    day_map = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    df['startDay'] = df['startDay'].map(day_map)
    df['endDay'] = df['endDay'].map(day_map)

        # Group by (id, id_2) pairs and check conditions
    result = df.groupby(['id', 'id_2']).apply(lambda x: (len(x) == 1) or (x['startDay'].nunique() == 7 and (x['endDay'].max() - x['startDay'].min()) == 6))
    
    return result
