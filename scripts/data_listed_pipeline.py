#required packages
import numpy as np
import pandas as pd
import chardet
import datetime



def join_listing_dates(df_wide, df_listings, df_key):


    #rename the column as it is not the correct property_id
    df_wide = df_wide.rename(columns={'property_id': 'random_property_id'})

    #add the cc_property_id to the df_wide dataframe
    df_joined_with_key = df_wide.merge(df_key, on='rex_property_id')

    #join the listing dataset to the expanded dataset using the property_id column
    df_listings = df_listings.rename(columns={'property_id': 'cc_property_id'})

    joined = df_listings.merge(df_joined_with_key, on='cc_property_id')

    
    joined["list_date"] = pd.to_datetime(joined["list_date"])
    joined["sale_date"] = pd.to_datetime(joined["sale_date"])

    joined.to_csv('../data/denver_joined_dataset.csv', index=False)

    return joined