
import psycopg2
import numpy
import pandas as pd
from scipy.stats import kurtosis,skew,variation,iqr,entropy,power_divergence,moment,jarque_bera
import numpy as np
import joblib
#from sklearn import linear_model
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import LinearSVC
#from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier

# setting a seed for reproducibility
numpy.random.seed(10)
# read all stock files in directory indivisual_stocks_5yr
'''
def read_all_stock_files(folder_path):
    allFiles = []
    for (_, _, files) in os.walk(folder_path):
        allFiles.extend(files)
        break

    dataframe_dict = {}
    for stock_file in allFiles:
        df = pd.read_csv(folder_path + "/" +stock_file)
        dataframe_dict[(stock_file.split('_'))[0]] = df

    return dataframe_dict
'''
def read_sample():
    sample_list=[i for i in range(6000)]
    return sample_list


def get_data_from_database(sample_id):
    # establishing the connection
    conn = psycopg2.connect(
        database="pelican", user='owais.ahmed@canarydetect.com', password='bLHshTcQ65fe8AnZ',
        host='canary-pelican-db-analytics-trials.cyeuxkq1eajh.us-east-1.rds.amazonaws.com', port='5432'
    )

    cursor = conn.cursor()

    query = 'select "id","channel","data","sample_id","gaussian_fit","type","baseline_gaussian" from sample_data where sample_id = {}'.format(sample_id)

    cursor.execute(query)
    sample_data_list = cursor.fetchall()

    df_raw = pd.DataFrame()
    for sample_data in sample_data_list:
        df1 = pd.DataFrame(sample_data[2], columns=[sample_data[1] + " " + sample_data[5]])
        df_raw = pd.concat([df_raw, df1], axis=1)

    df_gaussian = pd.DataFrame()
    for sample_data in sample_data_list:
        df1 = pd.DataFrame(sample_data[4], columns=[sample_data[1] + " " + sample_data[5] + " Gaussian"])
        df_gaussian = pd.concat([df_gaussian, df1], axis=1)

    df_gaussian_baseline = pd.DataFrame()
    for sample_data in sample_data_list:
        # print(len(sample_data[6]))
        df1 = pd.DataFrame(sample_data[6], columns=[sample_data[1] + " " + sample_data[5] + " Baseline Gaussian"])
        df_gaussian_baseline = pd.concat([df_gaussian_baseline, df1], axis=1)

        # Closing the connection
    conn.close()
    df = pd.concat([df_raw, df_gaussian, df_gaussian_baseline], axis=1, )
    df = df.reindex(
        ['C1 control', 'C1 control Gaussian', 'C1 control Baseline Gaussian', 'C2 control', 'C2 control Gaussian',
         'C2 control Baseline Gaussian', 'C1 test', 'C1 test Gaussian', 'C1 test Baseline Gaussian', 'C2 test',
         'C2 test Gaussian', 'C2 test Baseline Gaussian'], axis=1)

    return df,sample_id

def data_check(df_sampleid_tuple):
    df,sample_id=df_sampleid_tuple
    error_code = None

    if df['C1 control'].min() < 0 or df['C2 control'].min() < 0 or df['C1 test'].min() < 0 or df['C2 test'].min() < 0:
        prediction = 'Inconclusive '
        error_code = 'Error 800 - Raw Data negative for any steps C1 or C2 channel For Sample Id ' + str(sample_id)

    elif df['C1 control Gaussian'].min() < 0 or df['C2 control Gaussian'].min() < 0 or df[
        'C1 test Gaussian'].min() < 0 or df['C2 test Gaussian'].min() < 0:
        prediction = 'Inconclusive'
        error_code = 'Error 400 - Gaussian generated negative for any steps C1 or C2 channel For Sample ID ' + str(sample_id)

    elif len(df['C1 control Gaussian'].dropna()) == 0 or len(df['C2 control Gaussian'].dropna()) == 0 or len(
            df['C1 test Gaussian'].dropna()) == 0 or len(df['C2 test Gaussian'].dropna()) == 0:
        prediction = 'Inconclusive'
        error_code = 'Error 500 - Sensor response 0 for any steps C1 or C2 channel For Sample ID ' + str(sample_id)

    elif len(df['C1 control Baseline Gaussian'].dropna()) < 501 or len(
            df['C2 control Baseline Gaussian'].dropna()) < 501 or len(
            df['C1 test Baseline Gaussian'].dropna()) < 501 or len(df['C2 test Baseline Gaussian'].dropna()) < 501:
        prediction = 'Inconclusive'
        error_code = 'Error 300 - Bluetooth connection lost For Sample ID ' + str(sample_id)


    elif len(df['C1 control'].dropna()) > 601 or len(df['C2 control'].dropna()) > 601 or len(
            df['C1 test'].dropna()) > 601 or len(df['C2 test'].dropna()) > 601:
        prediction = 'Inconclusive'
        error_code = 'Error 200 - Timer Extended for any steps For Sample Id ' + str(sample_id)

    for i in df.loc[:, ["C1 control Baseline Gaussian", "C2 control Baseline Gaussian", "C1 test Baseline Gaussian",
                        "C2 test Baseline Gaussian"]].columns:
        max_after_420 = df[i].iloc[420:].max()
        max_before_420 = df[i].iloc[:420].max()
        if max_after_420 > max_before_420:
            prediction = 'Inconclusive'
            error_code = "Error 700 - Gauss Index greater than 420 For " + i + " For Sample ID " + str(sample_id)

    return df, error_code

    # convert an array of values into a dataset matrix


def feature_extraction(df,sampleid):

    feature_C1 = []
    feature_C2 = []

    minValue_C1_control = df['C1 control Baseline Gaussian'].min()
    maxValue_C1_control = df['C1 control Baseline Gaussian'].max()
    Peak_height_C1_control = maxValue_C1_control - minValue_C1_control
    peak_time_C1_control = df['C1 control Baseline Gaussian'].idxmax()
    area_C1_control = np.trapz(df.dropna()['C1 control Baseline Gaussian'],
                               x=(df.dropna()['C1 control Baseline Gaussian'].index))
    Kurtosis_C1_control = kurtosis(df.dropna()['C1 control Baseline Gaussian'].dropna())
    Skew_C1_control = skew(df.dropna()['C1 control Baseline Gaussian'].dropna())
    Variation_C1_control = variation(df['C1 control Baseline Gaussian'].dropna())
    Mean_Absolute_Deviation_C1_control = df['C1 control Baseline Gaussian'].dropna().mad()
    Percentile5th_C1_control = np.percentile(df['C1 control Baseline Gaussian'].dropna(), 5)
    Percentile95th_C1_control = np.percentile(df['C1 control Baseline Gaussian'].dropna(), 95)
    IQR_C1_control = iqr(df['C1 control Baseline Gaussian'].dropna())
    Entropy_C1_control = entropy(abs(df['C1 control Baseline Gaussian'].dropna()))

    Jarque_bera_C1_control = jarque_bera(df['C1 control Baseline Gaussian'].dropna())
    Jarque_bera_statistic_C1_control = Jarque_bera_C1_control[0]
    Jarque_bera_pvalue_C1_control = Jarque_bera_C1_control[1]

    Moment_C1_control = moment(df['C1 control Baseline Gaussian'].dropna(), moment=2)
    Mean_C1_control = df['C1 control Baseline Gaussian'].dropna().mean()
    Std_C1_control = df['C1 control Baseline Gaussian'].dropna().std()
    Coeff_of_variation_C1_control = (
                Std_C1_control / Mean_C1_control * 100)  # Coefficient of Variation = (Standard Deviation / Mean) * 100.

    minValue_C1_test = df['C1 test Baseline Gaussian'].min()
    maxValue_C1_test = df['C1 test Baseline Gaussian'].max()
    Peak_height_C1_test = maxValue_C1_test - minValue_C1_test
    peak_time_C1_test = df['C1 test Baseline Gaussian'].idxmax()
    area_C1_test = np.trapz(df.dropna()['C1 test Baseline Gaussian'],
                            x=(df.dropna()['C1 test Baseline Gaussian'].index))
    Kurtosis_C1_test = kurtosis(df.dropna()['C1 test Baseline Gaussian'].dropna())
    Skew_C1_test = skew(df.dropna()['C1 test Baseline Gaussian'].dropna())
    Variation_C1_test = variation(df['C1 test Baseline Gaussian'].dropna())
    Mean_Absolute_Deviation_C1_test = df['C1 test Baseline Gaussian'].dropna().mad()
    Percentile5th_C1_test = np.percentile(df['C1 test Baseline Gaussian'].dropna(), 5)
    Percentile95th_C1_test = np.percentile(df['C1 test Baseline Gaussian'].dropna(), 95)
    IQR_C1_test = iqr(df['C1 test Baseline Gaussian'].dropna())
    Entropy_C1_test = entropy(abs(df['C1 test Baseline Gaussian'].dropna()))

    Jarque_bera_C1_test = jarque_bera(df['C1 test Baseline Gaussian'].dropna())
    Jarque_bera_statistic_C1_test = Jarque_bera_C1_test[0]
    Jarque_bera_pvalue_C1_test = Jarque_bera_C1_test[1]

    Moment_C1_test = moment(df['C1 test Baseline Gaussian'].dropna(), moment=2)
    Mean_C1_test = df['C1 test Baseline Gaussian'].dropna().mean()
    Std_C1_test = df['C1 test Baseline Gaussian'].dropna().std()
    Coeff_of_variation_C1_test = (Std_C1_test / Mean_C1_test * 100)

    minValue_C2_control = df['C2 control Baseline Gaussian'].min()
    maxValue_C2_control = df['C2 control Baseline Gaussian'].max()
    Peak_height_C2_control = maxValue_C2_control - minValue_C2_control
    peak_time_C2_control = df['C2 control Baseline Gaussian'].idxmax()
    area_C2_control = np.trapz(df.dropna()['C2 control Baseline Gaussian'],
                               x=(df.dropna()['C2 control Baseline Gaussian'].index))
    Kurtosis_C2_control = kurtosis(df.dropna()['C2 control Baseline Gaussian'].dropna())
    Skew_C2_control = skew(df.dropna()['C2 control Baseline Gaussian'].dropna())
    Variation_C2_control = variation(df['C2 control Baseline Gaussian'].dropna())
    Mean_Absolute_Deviation_C2_control = df['C2 control Baseline Gaussian'].dropna().mad()
    Percentile5th_C2_control = np.percentile(df['C2 control Baseline Gaussian'].dropna(), 5)
    Percentile95th_C2_control = np.percentile(df['C2 control Baseline Gaussian'].dropna(), 95)
    IQR_C2_control = iqr(df['C2 control Baseline Gaussian'].dropna())
    Entropy_C2_control = entropy((df['C2 control Baseline Gaussian'].dropna()))

    Jarque_bera_C2_control = jarque_bera(df['C2 control Baseline Gaussian'].dropna())
    Jarque_bera_statistic_C2_control = Jarque_bera_C2_control[0]
    Jarque_bera_pvalue_C2_control = Jarque_bera_C2_control[1]

    Moment_C2_control = moment(df['C2 control Baseline Gaussian'].dropna(), moment=2)
    Mean_C2_control = df['C2 control Baseline Gaussian'].dropna().mean()
    Std_C2_control = df['C2 control Baseline Gaussian'].dropna().std()
    Coeff_of_variation_C2_control = (
                Std_C2_control / Mean_C2_control * 100)  # Coefficient of Variation = (Standard Deviation / Mean) * 100.

    minValue_C2_test = df['C2 test Baseline Gaussian'].min()
    maxValue_C2_test = df['C2 test Baseline Gaussian'].max()
    Peak_height_C2_test = maxValue_C2_test - minValue_C2_test
    peak_time_C2_test = df['C2 test Baseline Gaussian'].idxmax()
    area_C2_test = np.trapz(df.dropna()['C2 test Baseline Gaussian'],
                            x=(df.dropna()['C2 test Baseline Gaussian'].index))
    Kurtosis_C2_test = kurtosis(df.dropna()['C2 test Baseline Gaussian'].dropna())
    Skew_C2_test = skew(df.dropna()['C2 test Baseline Gaussian'].dropna())
    Variation_C2_test = variation(df['C2 test Baseline Gaussian'].dropna())
    Mean_Absolute_Deviation_C2_test = df['C2 test Baseline Gaussian'].dropna().mad()
    Percentile5th_C2_test = np.percentile(df['C2 test Baseline Gaussian'].dropna(), 5)
    Percentile95th_C2_test = np.percentile(df['C2 test Baseline Gaussian'].dropna(), 95)
    IQR_C2_test = iqr(df['C2 test Baseline Gaussian'].dropna())
    Entropy_C2_test = entropy(abs(df['C2 test Baseline Gaussian'].dropna()))

    Jarque_bera_C2_test = jarque_bera(df['C2 test Baseline Gaussian'].dropna())
    Jarque_bera_statistic_C2_test = Jarque_bera_C2_test[0]
    Jarque_bera_pvalue_C2_test = Jarque_bera_C2_test[1]

    Moment_C2_test = moment(df['C2 test Baseline Gaussian'].dropna(), moment=2)
    Mean_C2_test = df['C2 test Baseline Gaussian'].dropna().mean()
    Std_C2_test = df['C2 test Baseline Gaussian'].dropna().std()
    Coeff_of_variation_C2_test = (Std_C2_test / Mean_C2_test * 100)

    feature_C1.append(
        [minValue_C1_control, maxValue_C1_control, Peak_height_C1_control, peak_time_C1_control, area_C1_control,
         Kurtosis_C1_control, Skew_C1_control, Variation_C1_control, Mean_Absolute_Deviation_C1_control,
         Percentile5th_C1_control, Percentile95th_C1_control, IQR_C1_control, Entropy_C1_control,
         Jarque_bera_statistic_C1_control, Jarque_bera_pvalue_C1_control, Moment_C1_control, Mean_C1_control,
         Std_C1_control, Coeff_of_variation_C1_control, minValue_C1_test, maxValue_C1_test, Peak_height_C1_test,
         peak_time_C1_test, area_C1_test, Kurtosis_C1_test, Skew_C1_test, Variation_C1_test,
         Mean_Absolute_Deviation_C1_test, Percentile5th_C1_test, Percentile95th_C1_test, IQR_C1_test, Entropy_C1_test,
         Jarque_bera_statistic_C1_test, Jarque_bera_pvalue_C1_test, Moment_C1_test, Mean_C1_test, Std_C1_test,
         Coeff_of_variation_C1_test])
    feature_dataset_C1 = pd.DataFrame(feature_C1,
                                      columns=['minValue_control', 'maxValue_control', 'Peak_height_control',
                                               'peak_time_control', 'area_control', 'Kurtosis_control', 'Skew_control',
                                               'Variation_control', 'Mean_Absolute_Deviation_control',
                                               'Percentile5th_control', 'Percentile95th_control', 'IQR_control',
                                               'Entropy_control', 'Jarque_bera_statistic_control',
                                               'Jarque_bera_pvalue_control', 'Moment_control', 'Mean_control',
                                               'Std_control', 'Coeff_of_variation_control', 'minValue_test',
                                               'maxValue_test', 'Peak_height_test', 'peak_time_test', 'area_test',
                                               'Kurtosis_test', 'Skew_test', 'Variation_test',
                                               'Mean_Absolute_Deviation_test', 'Percentile5th_test',
                                               'Percentile95th_test', 'IQR_test', 'Entropy_test',
                                               'Jarque_bera_statistic_test', 'Jarque_bera_pvalue_test', 'Moment_test',
                                               'Mean_test', 'Std_test', 'Coeff_of_variation_test'])
    feature_dataset_C1.insert(0, 'Sample Id', sampleid)
    feature_C2.append(
        [minValue_C2_control, maxValue_C2_control, Peak_height_C2_control, peak_time_C2_control, area_C2_control,
         Kurtosis_C2_control, Skew_C2_control, Variation_C2_control, Mean_Absolute_Deviation_C2_control,
         Percentile5th_C2_control, Percentile95th_C2_control, IQR_C2_control, Entropy_C2_control,
         Jarque_bera_statistic_C2_control, Jarque_bera_pvalue_C2_control, Moment_C2_control, Mean_C2_control,
         Std_C2_control, Coeff_of_variation_C2_control, minValue_C2_test, maxValue_C2_test, Peak_height_C2_test,
         peak_time_C2_test, area_C2_test, Kurtosis_C2_test, Skew_C2_test, Variation_C2_test,
         Mean_Absolute_Deviation_C2_test, Percentile5th_C2_test, Percentile95th_C2_test, IQR_C2_test, Entropy_C2_test,
         Jarque_bera_statistic_C2_test, Jarque_bera_pvalue_C2_test, Moment_C2_test, Mean_C2_test, Std_C2_test,
         Coeff_of_variation_C2_test])
    feature_dataset_C2 = pd.DataFrame(feature_C2,
                                      columns=['minValue_control', 'maxValue_control', 'Peak_height_control',
                                               'peak_time_control', 'area_control', 'Kurtosis_control', 'Skew_control',
                                               'Variation_control', 'Mean_Absolute_Deviation_control',
                                               'Percentile5th_control', 'Percentile95th_control', 'IQR_control',
                                               'Entropy_control', 'Jarque_bera_statistic_control',
                                               'Jarque_bera_pvalue_control', 'Moment_control', 'Mean_control',
                                               'Std_control', 'Coeff_of_variation_control', 'minValue_test',
                                               'maxValue_test', 'Peak_height_test', 'peak_time_test', 'area_test',
                                               'Kurtosis_test', 'Skew_test', 'Variation_test',
                                               'Mean_Absolute_Deviation_test', 'Percentile5th_test',
                                               'Percentile95th_test', 'IQR_test', 'Entropy_test',
                                               'Jarque_bera_statistic_test', 'Jarque_bera_pvalue_test', 'Moment_test',
                                               'Mean_test', 'Std_test', 'Coeff_of_variation_test'])
    feature_dataset_C2.insert(0, 'Sample Id', sampleid)
    feature_dataset = pd.concat([feature_dataset_C1, feature_dataset_C2])
    feature_dataset = feature_dataset.replace(-np.inf, 1000)
    feature_dataset

    return feature_dataset

# Model Predictions functions

def linear_model_predict(feature_df):
    filename = 'Models/Logistic_Regression_with_Grid_Search.sav'
    model = joblib.load(filename)
    Predictions = model.predict(feature_df.drop(columns=['Sample Id']))
    Predictions_probability = model.predict_proba(feature_df.drop(columns=['Sample Id']))
    Prediction_C1 = Predictions[0]
    Prediction_C2 = Predictions[1]

    if Prediction_C1 == 0:
        prediction_probability_C1 = Predictions_probability[0][0]
        Prediction_C1 = 'Negative'

    elif Prediction_C1 == 1:
        prediction_probability_C1 = Predictions_probability[0][1]
        Prediction_C1 = 'Positive'

    if Prediction_C2 == 0:
        prediction_probability_C2 = Predictions_probability[1][0]
        Prediction_C2 = 'Negative'
    elif Prediction_C2 == 1:
        prediction_probability_C2 = Predictions_probability[1][1]
        Prediction_C2 = 'Positive'

    return [Prediction_C1, prediction_probability_C1], [Prediction_C2, prediction_probability_C2]

def KNeighborsClassifier_predict(feature_df):
        filename = 'Models/KNN_with_Grid_Search.sav'
        model = joblib.load(filename)
        Predictions = model.predict(feature_df.drop(columns=['Sample Id']))
        Predictions_probability = model.predict_proba(feature_df.drop(columns=['Sample Id']))
        Prediction_C1 = Predictions[0]
        Prediction_C2 = Predictions[1]

        if Prediction_C1 == 0:
            prediction_probability_C1 = Predictions_probability[0][0]
            Prediction_C1 = 'Negative'

        elif Prediction_C1 == 1:
            prediction_probability_C1 = Predictions_probability[0][1]
            Prediction_C1 = 'Positive'

        if Prediction_C2 == 0:
            prediction_probability_C2 = Predictions_probability[1][0]
            Prediction_C2 = 'Negative'
        elif Prediction_C2 == 1:
            prediction_probability_C2 = Predictions_probability[1][1]
            Prediction_C2 = 'Positive'

        return [Prediction_C1, prediction_probability_C1], [Prediction_C2, prediction_probability_C2]


def LinearSVC_predict(feature_df):
    filename = 'Models/Linear_SVC_with_Grid_Search.sav'
    model = joblib.load(filename)
    Predictions = model.predict(feature_df.drop(columns=['Sample Id']))
    Predictions_probability = model.predict_proba(feature_df.drop(columns=['Sample Id']))
    Prediction_C1 = Predictions[0]
    Prediction_C2 = Predictions[1]

    if Prediction_C1 == 0:
        prediction_probability_C1 = Predictions_probability[0][0]
        Prediction_C1 = 'Negative'

    elif Prediction_C1 == 1:
        prediction_probability_C1 = Predictions_probability[0][1]
        Prediction_C1 = 'Positive'

    if Prediction_C2 == 0:
        prediction_probability_C2 = Predictions_probability[1][0]
        Prediction_C2 = 'Negative'
    elif Prediction_C2 == 1:
        prediction_probability_C2 = Predictions_probability[1][1]
        Prediction_C2 = 'Positive'

    return [Prediction_C1, prediction_probability_C1], [Prediction_C2, prediction_probability_C2]


def kernalSVC_predict(feature_df):
    filename = 'Models/Kernel_SVM_with_Grid_Search.sav'
    model = joblib.load(filename)
    Predictions = model.predict(feature_df.drop(columns=['Sample Id']))
    Predictions_probability = model.predict_proba(feature_df.drop(columns=['Sample Id']))
    Prediction_C1 = Predictions[0]
    Prediction_C2 = Predictions[1]

    if Prediction_C1 == 0:
        prediction_probability_C1 = Predictions_probability[0][0]
        Prediction_C1 = 'Negative'

    elif Prediction_C1 == 1:
        prediction_probability_C1 = Predictions_probability[0][1]
        Prediction_C1 = 'Positive'

    if Prediction_C2 == 0:
        prediction_probability_C2 = Predictions_probability[1][0]
        Prediction_C2 = 'Negative'
    elif Prediction_C2 == 1:
        prediction_probability_C2 = Predictions_probability[1][1]
        Prediction_C2 = 'Positive'

    return [Prediction_C1, prediction_probability_C1], [Prediction_C2, prediction_probability_C2]


def DecisionTreeClassifier_predict(feature_df):
    filename = 'Models/Decision_Trees_with_Grid_Search.sav'
    model = joblib.load(filename)
    Predictions = model.predict(feature_df.drop(columns=['Sample Id']))
    Predictions_probability = model.predict_proba(feature_df.drop(columns=['Sample Id']))
    Prediction_C1 = Predictions[0]
    Prediction_C2 = Predictions[1]

    if Prediction_C1 == 0:
        prediction_probability_C1 = Predictions_probability[0][0]
        Prediction_C1 = 'Negative'

    elif Prediction_C1 == 1:
        prediction_probability_C1 = Predictions_probability[0][1]
        Prediction_C1 = 'Positive'

    if Prediction_C2 == 0:
        prediction_probability_C2 = Predictions_probability[1][0]
        Prediction_C2 = 'Negative'
    elif Prediction_C2 == 1:
        prediction_probability_C2 = Predictions_probability[1][1]
        Prediction_C2 = 'Positive'

    return [Prediction_C1, prediction_probability_C1], [Prediction_C2, prediction_probability_C2]



def RandomForestClassifier_predict(feature_df):
    filename = 'Models/Random_Forest_Classifier_with_Grid_Search.sav'
    model = joblib.load(filename)
    Predictions = model.predict(feature_df.drop(columns=['Sample Id']))
    Predictions_probability = model.predict_proba(feature_df.drop(columns=['Sample Id']))
    Prediction_C1 = Predictions[0]
    Prediction_C2 = Predictions[1]

    if Prediction_C1 == 0:
        prediction_probability_C1 = Predictions_probability[0][0]
        Prediction_C1 = 'Negative'

    elif Prediction_C1 == 1:
        prediction_probability_C1 = Predictions_probability[0][1]
        Prediction_C1 = 'Positive'

    if Prediction_C2 == 0:
        prediction_probability_C2 = Predictions_probability[1][0]
        Prediction_C2 = 'Negative'
    elif Prediction_C2 == 1:
        prediction_probability_C2 = Predictions_probability[1][1]
        Prediction_C2 = 'Positive'

    return [Prediction_C1, prediction_probability_C1], [Prediction_C2, prediction_probability_C2]
