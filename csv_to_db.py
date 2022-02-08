# import libraries
from csv_import_functions import *

# settings
dataset_dir = 'csv files'

# database credentials
host = '******.rds.amazonaws.com'
dbname = '******'
user = '******'
password = '******'

# configure environment and create main df
csv_files = csv_files()
df = create_df(dataset_dir, csv_files)

for k in csv_files:
    # call dataframe
    dataframe = df[k]

    # clean table name
    tbl_name = clean_tbl_name(k)

    # clean column names
    col_str, dataframe.columns = clean_colname(dataframe)

    # upload data to db
    upload_to_db(host,
                 dbname,
                 user,
                 password,
                 tbl_name,
                 col_str,
                 file='csv files/'+k,
                 dataframe=dataframe,
                 dataframe_columns=dataframe.columns)
