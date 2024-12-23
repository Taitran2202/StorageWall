
import joblib
import numpy as np
import pandas as pd
import logging

# Create helper function to create features based on smoothing the time series for features by adding rolling mean and rolling standard deviation
class Predictive_Maintenance:
   def __init__(self) -> None:
      self.features =  ['cycle', 'retry_clamp', 'retry_release', 'life_time', 'drop_decive', 'clamp_elapse_time', 'release_elapse_time']
      self.sequence_cols_X = ['retry_clamp', 'retry_release', 'life_time', 'drop_decive', 'clamp_elapse_time', 'release_elapse_time']
      self.sequence_cols_Y = ['need_maintenance']

   def add_features(self, df_in, rolling_win_size):
      """Add rolling average and rolling standard deviation for features readings using fixed rolling window size.
      Args:
         df_in (dataframe)     : The input dataframe to be proccessed (training or test) 
         rolling_win_size (int): The window size, number of cycles for applying the rolling function
      Reurns:
         dataframe: contains the input dataframe with additional rolling mean and std for each feature
      """
      features_cal = self.sequence_cols_X
      feature_av_cols = ["".join(['av_', nm]) for nm in self.sequence_cols_X]
      feature_sd_cols = ["".join(['sd_', nm]) for nm in self.sequence_cols_X]

      df_out = pd.DataFrame()

      ws = rolling_win_size

      df_sub = df_in[features_cal]

      av = df_sub.rolling(ws, min_periods=1).mean()
      av.columns = feature_av_cols

      sd = df_sub.rolling(ws, min_periods=1).std().fillna(0)
      sd.columns = feature_sd_cols

      df_out = pd.concat([df_in,av,sd], axis=1)
      
      features_cols_cal = features_cal + feature_av_cols + feature_sd_cols

      return df_out, features_cols_cal

   def gen_sequence(self, data, seq_length, seq_cols):
      data_matrix = data[seq_cols].values
      num_elements = data_matrix.shape[0]
      for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
         yield data_matrix[start:stop, :]

   def gen_labels(self, data, seq_length, label):
      data_matrix = data[label].values
      num_elements = data_matrix.shape[0]
      return data_matrix[seq_length:num_elements, :]

   def prepare_sequence_data(self, data, sequence_length, sequence_cols_X):

      seq_gen = list(self.gen_sequence(data, sequence_length, sequence_cols_X))
      seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
      seq_array = seq_array.reshape(len(data)-sequence_length, sequence_length, len(sequence_cols_X))
      print(seq_array.shape)
      print(seq_array)

      return seq_array

   
   def predict_bin_classify(self, model, X_test):

      """Perfor Grid Search hyper parameter tuning on a classifier.
      Args:
         model (str): The model name identifier
         clf (clssifier object): The classifier to be tuned
         features (list): The set of input features names
         params (dict): Grid Search parameters
         score (str): Grid Search score
      Returns:
         Tuned Clssifier object
         dataframe of model predictions and scores
      """

      grid_search = joblib.load(f'/home/greystone/StorageWall/model_template/PredictMaintenance/{model}.joblib')
      y_pred = grid_search.predict(X_test)

      if hasattr(grid_search, 'predict_proba'):   
         y_score = grid_search.predict_proba(X_test)[:,1]
      elif hasattr(grid_search, 'decision_function'):
         y_score = grid_search.decision_function(X_test)
      else:
         y_score = y_pred

      return y_pred, y_score
   
   def run(self, csv_file_path):
      #df_train = pd.read_excel('/home/greystone/StorageWall/data_debug/train.xlsx',header = None)
      result_string = None
      score_string = None
      try:
         data_origin = pd.read_csv(csv_file_path,header = None)
         df_data_origin = pd.DataFrame(data_origin)
         df_data_origin = df_data_origin.astype('float64')
         df_data_origin.drop([0], axis=1, inplace=True)
         print(df_data_origin.dtypes)
         
         print(df_data_origin)

         # __In model names:__  
         # 
         # __B__ stands for applying the model on the original features set, __B__efore feature extraction  
         # __A__ stands for applying the model on the original + extracted features set, __A__fter feature extraction  

         # ### SVC Linear
         #model = 'SVC_Linear_B'
         # model = 'SVC_Linear_A'
         # ### Logistic Regression
         #model = 'Logistic_Regression_B'
         # model = 'Logistic_Regression_A'
         # ### Gaussian Naive Bayes
         # model = 'Gaussian_NB_B'
         # model = 'Gaussian_NB_A'
         # ### Random forest
         # model = 'Random_Forest_B'
         # model = 'Random_Forest_A'
         # ### KNeighbors
         # model = 'KNN_B'
         # model = 'KNN_A'
         # ### Support vector machine
         model = 'SVC_B'
         # model = 'SVC_A'
         
         data_input = (np.array(df_data_origin).flatten()).reshape(1,-1)
         
         print(data_input.shape)
         
         result, score = self.predict_bin_classify(model,data_input)

         if result == 0:
            result_string = 'no_need_maintenance'
         elif result == 1:
            result_string = 'need_maintenance'
      
         return result_string, score
      except Exception as e:
         logging.exception('Exception in Predictive_Maintenance: {}'.format(e))
      

if __name__ == '__main__':
   csv_file_path = '/home/greystone/StorageWall/data_debug/test.csv'
   
   model = Predictive_Maintenance()
   result_string, score = model.run(csv_file_path)
   
   print('result',result_string)
   print('score',score)
