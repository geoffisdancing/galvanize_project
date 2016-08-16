

def load_clean_HF():
    '''
    INPUT: Requires two files 'meas_fitbit_tracker.txt' and 'surv_medical_conditions.txt' to be in the working dir and will return
    test train split vars for the HF outcome to use in modeling, after cleaning and subsetting the variables
    OUTPUT: test train split vars for the HF outcome
    '''
    #load fitbit data
    fit = pd.read_table('meas_fitbit_tracker.txt', sep='|', parse_dates=[1])
    fit['date']=pd.to_datetime(fit['date'], format='%m/%d/%Y') #convert fitbit date to datetime
    fit = fit.drop(['distance', 'calories', 'floors', 'elevation'], axis=1)
    fit['steps_zero'] = fit.steps
    fit['steps_zero'] = fit['steps_zero'][fit['steps'].isnull()==True]=0
    fit['steps_missing'] = fit.steps
    fit['steps_missing'] = fit['steps_missing'][fit['steps']==0]=float('NaN')
    #now create a days used column for the entire fit df
    fit['days_used']=fit['date'].groupby(fit['user_id']).transform('count')

    #I'm going to pause here to see if I can pivot to building a recurrent neural net with the Fitbit data.

    ##here i will pause to try to run a recurrent neural network on daily data.
    #so what i need to do is to create an X df with the last 60d of people's step data
    #ranging from 0 to x (convert NaN to 0)
    #and a Y df with blockages in coroary vs. not
    # so start by creating datasets according to these parameters

    # First data-clean: drop users with days_used < 180
    fit_clean = fit
    fit_clean.drop(fit_clean[fit_clean.days_used<180].index, inplace=True)
    #Now to create a dataset with only the last 60d of data for people
    #start by calculating the max date for everyone and dropping data older than 60 than each person's last date
    fit_clean['max_date']=fit_clean['date'].groupby(fit_clean['user_id']).transform('max')
    fit_clean.drop(fit_clean[fit_clean.date<(fit_clean.max_date-timedelta(days=59))].index, inplace=True)
    fit_clean.steps.fillna(0, inplace=True) #replace NaN steps with 0 for neural network

    #read med_cond into df and create a mini df with only desired columns
    med_cond = pd.read_table('surv_medical_conditions.txt', sep='|')
    mini_med_cond = med_cond[['user_id', 'hbp', 'high_chol', 'diabetes', 'blockages_in_your_coronary', 'heart_attack', 'pvd', 'clots', 'chf', 'stroke', 'enlarged', 'afib', 'arrhythmia', 'murmur', 'valve', 'congenital' ,'pulm_hyper', 'aorta', 'sleep_apnea', 'copd', 'asthma', 'arrhythmia1', 'arrhythmia2']]


    #now merge fit and mini_med_cond df's
    union_med = pd.merge(fit_clean, mini_med_cond, how='left', on='user_id', copy=True)
    X_hf = union_med[union_med.chf==1]
    X_hf_control = union_med[union_med.chf==2]
    control_user_ids = X_hf_control.user_id.unique()

    X_hf_control_user_list = random.sample(control_user_ids, __ )
    X_hf_control = X_hf_control[X_hf_control['user_id'].isin(X_hf_control_user_list)]
    #So now merging X_hf and X_hf_control into the X variable
    X_hf_full = pd.concat([X_hf, X_hf_control])
    #resetting index of X_cad_full and removing unneeded variable
    X_hf_full.reset_index(inplace=True)
    X_hf_full = X_hf_full.drop(['index', 'level_0'], axis=1)
    # now need to restructure data to be one row per user, and 60 columns wide
    sixty=pd.DataFrame([np.tile(np.arange(0,60), 332)]).T ##creating a df with repeating 0-59 to denote each person in df.pivot
    sixty.columns=['ind'] #naming new sequence variable so we can reference it in the pivot method
    X_hf_model = X_hf_full.join(sixty)
    X_hf_model2 = X_hf_model[['user_id', 'steps', 'ind']] #selecting only those columns needed
    #pivoting df to create 1 row per user and 60 features wide (1 feature per daily step count)
    X_hf_model_arr = X_hf_model2.pivot(index='user_id', values='steps', columns='ind')
    #resetting index and removing user_id variable
    X_hf_model_arr.reset_index(inplace=True)
    X_hf_model_arr.drop('user_id', axis=1, inplace=True)
    #"X_hf_model_arr" IS THE PREDICTOR VAR TO USE IN THE RNN MODEL

    #create response variable Y
    Y_labels = X_hf_full.groupby('user_id').mean()['chf']
    #So it looks like there are __ NaNs and __ "3" values in the Y_labels, so I will turn both of these to "2"
    Y_labels.replace(to_replace=3, value=2, inplace=True)
    Y_labels.fillna(value=2, inplace=True)
    Y_labels.replace(to_replace=2, value=0, inplace=True) #changing no CAD to 0 from 2
    Y = Y_labels.values #FINAL TARGET VAR TO USE IN THE RNN

    X_train, X_test, Y_train, Y_test = train_test_split(X_cad_model_arr, Y, test_size=0.2)

    #now, last thing, we have to format the X_train/test vars as keras wants them for RNN
    x_train = X_train.values[:,:,None]
    x_test = X_test.values[:,:,None]

    return x_train, x_test, Y_train, Y_test




def load_clean_custom(disease_var):
    '''
    INPUT: Requires two files 'meas_fitbit_tracker.txt' and 'surv_medical_conditions.txt' to be in the working dir and will return
    test train split vars for the disease_var outcome to use in modeling, after cleaning and subsetting the variables
    OUTPUT: test train split vars for the disease_var outcome
    '''

    #load fitbit data
    fit = pd.read_table('meas_fitbit_tracker.txt', sep='|', parse_dates=[1])
    fit['date']=pd.to_datetime(fit['date'], format='%m/%d/%Y') #convert fitbit date to datetime
    fit = fit.drop(['distance', 'calories', 'floors', 'elevation'], axis=1)
    fit['steps_zero'] = fit.steps
    fit['steps_zero'] = fit['steps_zero'][fit['steps'].isnull()==True]=0
    fit['steps_missing'] = fit.steps
    fit['steps_missing'] = fit['steps_missing'][fit['steps']==0]=float('NaN')
    #now create a days used column for the entire fit df
    fit['days_used']=fit['date'].groupby(fit['user_id']).transform('count')

    #I'm going to pause here to see if I can pivot to building a recurrent neural net with the Fitbit data.

    ##here i will pause to try to run a recurrent neural network on daily data.
    #so what i need to do is to create an X df with the last 60d of people's step data
    #ranging from 0 to x (convert NaN to 0)
    #and a Y df with blockages in coroary vs. not
    # so start by creating datasets according to these parameters

    # First data-clean: drop users with days_used < 180
    fit_clean = fit
    fit_clean.drop(fit_clean[fit_clean.days_used<180].index, inplace=True)
    #Now to create a dataset with only the last 60d of data for people
    #start by calculating the max date for everyone and dropping data older than 60 than each person's last date
    fit_clean['max_date']=fit_clean['date'].groupby(fit_clean['user_id']).transform('max')
    fit_clean.drop(fit_clean[fit_clean.date<(fit_clean.max_date-timedelta(days=59))].index, inplace=True)
    fit_clean.steps.fillna(0, inplace=True) #replace NaN steps with 0 for neural network

    #read med_cond into df and create a mini df with only desired columns
    med_cond = pd.read_table('surv_medical_conditions.txt', sep='|')
    mini_med_cond = med_cond[['user_id', 'hbp', 'high_chol', 'diabetes', 'blockages_in_your_coronary', 'heart_attack', 'pvd', 'clots', 'chf', 'stroke', 'enlarged', 'afib', 'arrhythmia', 'murmur', 'valve', 'congenital' ,'pulm_hyper', 'aorta', 'sleep_apnea', 'copd', 'asthma', 'arrhythmia1', 'arrhythmia2']]

    #now merge fit and mini_med_cond df's
    union_med = pd.merge(fit_clean, mini_med_cond, how='left', on='user_id', copy=True)
    X_disease = union_med[union_med[disease_var]==1]
    X_disease_control = union_med[union_med[disease_var]==2]
    control_user_ids = X_disease_control.user_id.unique()

    case_num = len(X_disease.user_id.unique())
    X_disease_control_user_list = random.sample(control_user_ids, case_num)
    X_disease_control = X_disease_control[X_disease_control['user_id'].isin(X_disease_control_user_list)]
    #So now merging X_disease and X_disease_control into the X variable
    X_disease_full = pd.concat([X_disease, X_disease_control])
    #resetting index of X_disease_full and removing unneeded variable
    X_disease_full.reset_index(inplace=True)
    X_disease_full = X_disease_full.drop(['index'], axis=1)
    # now need to restructure data to be one row per user, and 60 columns wide
    sixty=pd.DataFrame([np.tile(np.arange(0,60), case_num*2)]).T ##creating a df with repeating 0-59 to denote each person in df.pivot
    sixty.columns=['ind'] #naming new sequence variable so we can reference it in the pivot method
    X_disease_model = X_disease_full.join(sixty)
    X_disease_model2 = X_disease_model[['user_id', 'steps', 'ind']] #selecting only those columns needed
    #pivoting df to create 1 row per user and 60 features wide (1 feature per daily step count)
    X_disease_model_arr = X_disease_model2.pivot(index='user_id', values='steps', columns='ind')
    #resetting index and removing user_id variable
    X_disease_model_arr.reset_index(inplace=True)
    X_disease_model_arr.drop('user_id', axis=1, inplace=True)
    #"X_disease_model_arr" IS THE PREDICTOR VAR TO USE IN THE RNN MODEL

    #create response variable Y
    Y_labels = X_disease_full.groupby('user_id').mean()[disease_var]
    #So it looks like there are 13 NaNs and 2 "3" values in the Y_labels, so I will turn both of these to "2"
    Y_labels.replace(to_replace=3, value=2, inplace=True)
    Y_labels.fillna(value=2, inplace=True)
    Y_labels.replace(to_replace=2, value=0, inplace=True) #changing no disease_var to 0 from 2
    Y = Y_labels.values #FINAL TARGET VAR TO USE IN THE RNN

    X_train, X_test, Y_train, Y_test = train_test_split(X_disease_model_arr, Y, test_size=0.2)

    #now, last thing, we have to format the X_train/test vars as keras wants them for RNN
    x_train = X_train.values[:,:,None]
    x_test = X_test.values[:,:,None]

    return x_train, x_test, Y_train, Y_test
