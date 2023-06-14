import csv


def fetch_column_as_tuple(csv_file_path, key_column_index=0, value_column_index=2):
    result_dict = {}
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            key = row[key_column_index].replace('.', '')
            value = f"{row[value_column_index]}.NS"
            result_dict[key] = value
    return result_dict


# Call the function to read the CSV file and create a tuple
csv_data = fetch_column_as_tuple('data/ind_nifty500list.csv', 0, 2)
