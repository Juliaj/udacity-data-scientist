"""Combine transcript, portfolio datasets and produce new dataset for offer response modeling.

The dataset covers offers with "offer received" event.
"""
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import sleep

import data_processing.util as util
import data_processing.transcript as dptrans

OFFER_NOT_RESPONDED = 'not_responded'
OFFER_RESPONSE_UNKNOWN = 'unknown'


def get_completion_purchase_amount(row, offers_df, transactions_df ):
    """Helper function to retrieve offer completion events and transaction made during offer period.
        
        In the scenario that mulitple offer received events were sent to customer, offer receive time is the time when 
        customer received the first offer event. The offer period is the time from the last offer received + offer duration

    Args:
        row: a record in a panda dataframe combined_df
        transcript_df: the proprocessed transcript data
    Returns:
        valid_offer_comp_cnt: (integer) valid number of offer completions 
        amount: (float) purchases made during offer period

    """
    customer_id, offer_id, offer_duration = row['customer_id'], row['offer_id'], row['duration']
    
    offer_received_df = offers_df[(offers_df['customer_id'] == customer_id) & (offers_df['offer_id'] == offer_id) & (offers_df['offer received'] == 1)]
    offer_completed_df = offers_df[(offers_df['customer_id'] == customer_id) & (offers_df['offer_id'] == offer_id) & (offers_df['offer completed'] == 1)]
    trans_df = transactions_df[transactions_df['customer_id'] == customer_id]

    offer_recv_time = offer_received_df['time'].values[0]
    offer_valid_utill = offer_received_df['time'].values[-1] + offer_duration

    valid_offer_comp = np.logical_and(offer_completed_df['time'] >= offer_recv_time, offer_completed_df['time'] <= offer_valid_utill)
    valid_offer_comp_cnt = valid_offer_comp.sum()

    valid_trans = trans_df[ (trans_df['time'] >= offer_recv_time) & (trans_df['time'] <= offer_valid_utill)]
    amount = valid_trans['amount'].sum().item()
    # print(f'offer_duration, offer_recv_time, offer_valid_utill = {offer_duration}, {offer_recv_time}, {offer_valid_utill}')
    return valid_offer_comp_cnt, amount

def get_offer_response(row):
    """Helper function to determine whether an offer is responded.

    Args:
        row: row of aggregated transcript dataframe
        offer_received_time: integer offer received time by a customer
        offer_period: integer offer valid period for a customer
    
    Returns:
        status: either not_responded or unknown

    """

    # offer never viewed
    if row['offer_viewed_sum'] == 0 :
        return OFFER_NOT_RESPONDED
    
    # bogo or discount offer never completed
    if (row['bogo'] == 1 or row['discount'] == 1) and row['offer_completed_sum'] == 0:
        return  OFFER_NOT_RESPONDED
    
    return OFFER_RESPONSE_UNKNOWN


def combine(portfolio_df, profile_df, transcript_df):
    """Create a combined dataframe from the transaction, demographic and offer data:
    
    Args:
        portfolio_df - a preprocessed panda dataframe contains offer metadata
        profile_df - a preprocessed panda dataframe contains customer demographic data
        transcript_df - a preprocessed panda dataframe contains transcript data
        
    Returns:
        combined_df - (dataframe),combined data from transaction, demographic and offer data
        
    """
    
    offers_df, transactions_df = dptrans.separate_offers_transactions(transcript_df)

    # aggregate transcript data by customer and offer
    aggd_df = dptrans.agg_offer_events(offers_df)
    print(f'aggd_df shape:{aggd_df.shape}')

    # add offer meta info to aggd
    combined_df = pd.merge(aggd_df, portfolio_df, on='offer_id', how='left')

    # capture customer's responses offers
    offer_response = []
    # transaction made within offer valid period
    tran_amount_in_period = []

    pbar = tqdm(total=aggd_df.shape[0])

    for i, row in combined_df.iterrows():  
        if i % 100 == 0:
            pbar.update(100)

        # print(f'person, customer_id, offer_id, (inf, d, b) = \'{row["customer_id"]}\', \'{row["customer_id"]}\',\'{row["offer_id"]}\', ({row["informational"]}, {row["discount"]}, {row["bogo"]})')

        if get_offer_response(row) == OFFER_NOT_RESPONDED:
            offer_response.append(0)
            tran_amount_in_period.append(0)
            # print(f'respond, amount, offer_recv_time, offer_valid_utill = 0, 0, 0, 0\n')     
        else:
            # look into offer completion, transctions made during offer period
            valid_offer_comp_cnt, amount = get_completion_purchase_amount(row, offers_df, transactions_df )
            if amount == 0:
                responded = 0
            elif row['informational'] == 0 and valid_offer_comp_cnt == 0:
                responded = 0
            else:
                responded = 1
  
            # add to collection
            offer_response.append(responded)
            amount = amount if responded == 1 else 0 
            tran_amount_in_period.append(amount)
            # print(f'respond, amount = {responded}, {amount}\n') 

    pbar.close()
    # add new data to combined_df
    combined_df['responded'] = offer_response
    combined_df['purchase_during_offer'] = tran_amount_in_period

    # add customer demographic info
    combined_df = pd.merge(combined_df, profile_df, on='customer_id', how='left')
    return combined_df


def map_offer_type(x, offer_types):
    """Used in lambda function.
    """
    for t in offer_types:
        if x[t] == 1:
            return t

def get_offer_response_summary(combined_df):
    '''Create dataset to summarize the success rate for each offer_type

    Args:
        combined_df: a panda dataframe combined profile, portfolio and transcript data

    Returns:
        offer_success_df: a pandas dataframe with the columns offer_id, offer_type, success_rate
    '''
    response_summary = combined_df.groupby(['offer_id']).agg(count=('responded', 'count'), responded_count=('responded', 'sum')).reset_index()
    response_summary['success_rate'] = response_summary.apply(lambda x: round(x['responded_count']/x['count']*100, 2), axis=1)
    response_summary = pd.merge(response_summary, combined_df[['offer_id', 'bogo', 'discount', 'informational']].drop_duplicates(subset=['offer_id']), on='offer_id', how='left')
    offer_types = ['bogo', 'discount', 'informational']
    response_summary['offer_type'] = response_summary.apply(lambda x: map_offer_type(x, offer_types), axis=1)
    response_summary.drop(offer_types, axis=1, inplace=True)
    response_summary=response_summary.sort_values(by=['success_rate'], ascending=False)

    return response_summary

def main():
    
    if len(sys.argv) == 5:

        transcript_pkl_file, profile_pkl_file, portfolio_pkl_file, output_filepath = sys.argv[1:]
        
        print(f'Loading transcript data from {transcript_pkl_file}.')
        transcript_df = util.load_pkl(transcript_pkl_file)
        print(f'    Input data shape: {transcript_df.shape}')

        print(f'Loading profile data from {profile_pkl_file}.')
        profile_df = util.load_pkl(profile_pkl_file)
        print(f'    Input data shape: {profile_df.shape}')

        print(f'Loading portfolio data from {portfolio_pkl_file}.')
        portfolio_df = util.load_pkl(portfolio_pkl_file)
        print(f'    Input data shape: {portfolio_df.shape}')
        
        print('Combining all datasets ...')
        combined_df = combine(portfolio_df, profile_df, transcript_df)
        print(f'    After transform, data shape: {combined_df.shape}')

        print(f'Saving data to {output_filepath}')
        util.save(combined_df, output_filepath)
        print('Data saved.')

        print('Creating offer response summary ...')
        offer_response = get_offer_response_summary(combined_df)
        output_summary_filepath = os.path.join(os.path.dirname(output_filepath), 'offer_summary.pkl')
        print(f'Saving data to {output_summary_filepath}')
        util.save(offer_response, output_summary_filepath)
        print("All done!")

    else:
        print(f'Please provide the input file path of profile data '\
              f'as well as the file path to save the cleaned data \n\nExample: python {os.path.basename(__file__)} '\
              f'../data/1_interim/transcript.pkl ../data/1_interim/profile.pkl ../data/1_interim/portfolio.pkl ../data/1_interim/offer_response.pkl')

if __name__ == '__main__':
    main()







