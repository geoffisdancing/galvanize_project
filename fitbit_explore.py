import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn

def exploratory_grouping():
    print "unique users: " , len(fit.user_id.unique())

    #summary of number of days of data per person
    len(fit.user_id.unique())
    user_ids = fit['user_id'].unique()
    mask = fit['user_id']==623
    user623 = fit[mask]
    plt.plot(np.arange(len(user623)), user623['steps'])
    user623['steps'].hist()
    plt.show()
    fit.groupby('user_id').count()['date'].describe() #number of days of data per person

def data_exploration():
    #Those with Fitbit n = 2250
    len(fit.user_id.unique())
    #Total pop with HF n = 1302
    len(med_cond[med_cond.chf==1].user_id.unique())

    #explore overlap of Fitbit with Medical conditions
    len(union_med[union_med.chf==1]['user_id'].unique()) #51
    len(union_med[union_med.hbp==1]['user_id'].unique()) #703
    len(union_med[union_med.diabetes==1]['user_id'].unique()) #151
    len(union_med[union_med.heart_attack==1]['user_id'].unique()) #97
    len(union_med[union_med.blockages_in_your_coronary==1]['user_id'].unique()) #167
    len(union_med[union_med.stroke==1]['user_id'].unique()) #55
    len(union_med[union_med.afib==1]['user_id'].unique()) #89
    len(union_med[union_med.copd==1]['user_id'].unique()) #53
    len(union_med[union_med.asthma==1]['user_id'].unique()) #205
    len(union_med[union_med.high_chol==1]['user_id'].unique()) #804
    #those that could be plausibly aided by prediction of using fibit data:
    # HF, copd, asthma. Most of the others are "silent"
    len(hf.user_id.unique()) #number of peopel who filled out the HF survey 1620 (larger than those who reported having HF)
    len(union_hf.user_id.unique()) #number of people with fitbits who filled out HF survey 91
    len(physact.user_id.unique()) #30021 unique users who filled out phys act
    len(union_pa.user_id.unique()) #***1835 users with fitbits who filled out physact survey, this could be interesting
    len(union_sob.user_id.unique()) #637 fitbit users with SOB surveys, out of 12319 who filled out SOB surveys to begin with
    union_sob.groupby('user_id').count()['started_at'].describe() #description of the mean number of sob surveys among fitbit users.
    # there's a mean of 1.5 and median of 1 survey per person
    len(sob_surv['user_id'].unique()) #184 folks with ginger SOB surveys
    len(union_sob_sover.user_id.unique()) #***2062 people with fitibit filled out a baseline SOB survey, which is good overlap
    union_sob_ging = pd.merge(sob_surv, fit_group, how='inner', on='user_id', copy=True)
    len(union_sob_ging['user_id'].unique()) #***54 unique Fitbit users with Ginger SOB surveys

    union_sob_ging.groupby('user_id').count()['date'].describe() #description of the mean number of ginger SOB surveys among Fitbit
    # mean 240 surveys per person, median 182 surveys, max 713 surveys. I wonder what the overlap is with the other SOB surveys.
    len(union_sob_2surv.user_id.unique()) #32 overlapping users with both HeH SOB and Ginger SOB surveys
    symp_over.shape # one row per user, there are 36742 respondants, and SOB is in there, so we likely have this in general on most fitibt folks
    sleep.shape # one row per user, 6191 respondents, not many overall (so likely minor feature), interesting components here are sleep hours, overall sleep quality.
    mood.groupby('user_id').sum().shape #25926 unique respondents for mood survey, interesting questions here: feelingdown, littleenergy, concentration
    fit_sob_ginger_day.shape #seems there are about 9592 days where ginger SOB survey and fitibt data overlap
    '''
    Looking at Mike Klein's feature importances, it seems countActive, MeanActive, stdActive, autocorrActive1, max1/10/60 were all predictive engineered features for SOB.
    It actually seems to make sense that rather than doing time-series on what will be fairly sparse data, what makes more sense is to engineer features that capture the
    density of activity from the minute to minute data. Certainly at least as a first pass.
    '''



if __name__ == '__main__':
    # Read Data
    fit = pd.read_table('/Data Files/2014-/Research/HeH/HeH Partners/Fitbit Analysi\
    s/Fitbit Data Analysis/160801 Galvanize Data/meas_fitbit_tracker.txt', sep='|', parse_dates=[1])
    fit_group = fit.groupby('user_id').sum() #grouping fitibt users, so one row per user_id, which will help count fitbit users with particular surveys
    fit_group.reset_index(inplace=True) #pulling index into 1st column to allow merging
    fit['date']=pd.to_datetime(fit['date'], format='%m/%d/%Y') #convert fitbit date to datetime
    minute = pd.read_table('meas_fitbit_intraday_1week.txt', sep='|', parse_dates=True)
    med_cond = pd.read_table('surv_medical_conditions.txt', sep='|')
    union_med = pd.merge(fit, med_cond, how='left', on='user_id', copy=True)
    hf = pd.read_table('surv_heart_failure.txt', sep='|', parse_dates=True)
    union_hf = pd.merge(hf, fit, how='inner', on='user_id', copy=True)
    physact = pd.read_table('surv_physact.txt', sep='|', parse_dates=True)
    union_pa = pd.merge(physact, fit_group, how='inner', on='user_id', copy=True)
    sob = pd.read_table('surv_sob.txt', sep='|', parse_dates=True)
    union_sob = pd.merge(sob, fit_group, how='inner', on='user_id', copy=True)
    ginger = pd.read_table('meas_ginger_surveys.txt', sep='|', parse_dates=True) #there are other SOB surveys buried in this Ginger dataset
    ginger['date']=pd.to_datetime(ginger['date'], format='%d%b%y:%H:%M:%S') #reformating date column into datetime object
    #below are some datetime maneuvers to get a date without time to merge with fitbit date format
    ginger['day']=pd.DatetimeIndex(ginger.date).normalize()
    ginger['date_full']=ginger['date']
    ginger['date'] = ginger['day']
    ginger.drop('day', axis=1, inplace=True)

    daily_symp=ginger[ginger['name']=='Daily Symptoms'] #interm daily Ginger symp df, can delete
    sob_surv = daily_symp[daily_symp['question']=='How much did your shortness of breath bother you YESTERDAY?'] #Ginger SOB survey
    union_sob_ging = pd.merge(sob_surv, fit_group, how='inner', on='user_id', copy=True) #merge between Ginger SOB survey and those with Fitbit
    union_sob_ging_days = pd.merge(sob_surv, fit, on=['user_id', ''])
    union_sob_2surv = pd.merge(union_sob_ging, sob, how='inner', on='user_id', copy=True) #merge between Ginger SOB survey and HeH SOB survey
    symp_over = pd.read_table('surv_symptom_overview.txt', sep='|', parse_dates=True) #symptoms overview dataset to explore
    sob_sover = symp_over[['user_id', 'sob1', 'sob_level1']] #baseline SOB survey asked in symptom overview survey n=36742
    union_sob_sover = pd.merge(sob_sover, fit_group, how='inner', on='user_id', copy=True) #merge between symp_overview SOB survey and those with Fitbit
    sleep = pd.read_table('surv_sleep_quality.txt', sep='|', parse_dates=True)
    mood = pd.read_table('surv_mood.txt', sep='|', parse_dates=True)
    fit_sob_ginger_day = pd.merge(fit, sob_surv, on=['user_id', 'date']) #note that based on Mike's notebook, you might need to subtract 1 from sob_surv to get "today"






    # sob_surv['date']=pd.to_datetime(sob_surv['date'], format='%d%b%y:%H:%M:%S') #reformating date column into datetime object
    # sob_surv['day']=pd.DatetimeIndex(sob_surv.date).normalize()
    # sob_surv['date_full']=sob_surv['date']
    # sob_surv['date'] = sob_surv['day']
    # sob_surv.drop('day', axis=1, inplace=True)
