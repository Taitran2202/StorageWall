import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('ggplot')

import joblib
from sklearn import metrics, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D


# Create helper function to create features based on smoothing the time series for features by adding rolling mean and rolling standard deviation
class Train_Predictive_Maintenance:
   def __init__(self) -> None:
      self.features =  ['ID', 'cycle', 'retry_clamp', 'retry_release', 'life_time', 'drop_decive', 'clamp_elapse_time', 'release_elapse_time', 'need_maintenance']
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

      for m_id in pd.unique(df_in.ID):
         df_engine = df_in[df_in['ID'] == m_id]

         df_sub = df_engine[features_cal]

         av = df_sub.rolling(ws, min_periods=1).mean()
         av.columns = feature_av_cols

         sd = df_sub.rolling(ws, min_periods=1).std().fillna(0)
         sd.columns = feature_sd_cols

         new_ftrs = pd.concat([df_engine,av,sd], axis=1)
         
         df_out = pd.concat([df_out,new_ftrs], axis=0)
      
      features_cols_cal = features_cal + feature_av_cols + feature_sd_cols

      return df_out, features_cols_cal

   def explore_col(self, df_out_train, s, e):
         
      """Plot 4 main graphs for a single feature.
      
         plot1: histogram 
         plot2: boxplot 
         plot3: line plot (time series over cycle)
         plot4: scatter plot vs. regression label ttf
         
      Args:
         s (str): The column name of the feature to be plotted.
         e (int): The number of random engines to be plotted for plot 3. Range from 1 -100, 0:all engines, >100: all engines.

      Returns:
         plots
      
      """
      
      fig = plt.figure(figsize=(15, 15))

      sub1 = fig.add_subplot(221) 
      sub1.set_title(s +' histogram')
      sub1.hist(df_out_train[s])

      # sub2 = fig.add_subplot(222)
      # sub2.set_title(s +' boxplot')
      # sub2.boxplot(df_out_train[s])
      
      #np.random.seed(12345)
      
      if e > 100 or e <= 0:
         select_engines = list(pd.unique(df_out_train.ID))
      else:
         select_engines = np.random.choice(range(1,11), e, replace=False)
         
      sub3 = fig.add_subplot(212)
      sub3.set_title('time series: ' + s +' / cycle')
      sub3.set_xlabel('cycle')
      for i in select_engines:
         df = df_out_train[['cycle', s]][df_out_train.ID == i]
         sub3.plot(df['cycle'],df[s])
      
      sub4 = fig.add_subplot(222)
      sub4.set_title("scatter: "+ s + " / need_maintenance (regr label)")
      sub4.set_xlabel('need_maintenance')
      sub4.scatter(df_out_train['need_maintenance'],df_out_train[s])


      plt.tight_layout()
      plt.savefig(f'/home/greystone/StorageWall/data_debug/Plot_debug/explore_{s}.png')
      sns.reset_orig()
      
   def plot_time_series(self, df_out_train, s):
       
      """Plot time series of a single sensor for 10 random sample engines.
      
         Args:
         s (str): The column name of the sensor to be plotted.

      Returns:
         plots
         
      """
      
      fig, axes = plt.subplots(10, 1, sharex=True, figsize = (15, 15))
      fig.suptitle(s + ' time series / cycle', fontsize=15)
      
      #np.random.seed(12345)
      select_engines = np.random.choice(range(1,11), 10, replace=False).tolist()
      
      for e_id in select_engines:
         df = df_out_train[['cycle', s]][df_out_train.ID == e_id]
         i = select_engines.index(e_id)
         axes[i].plot(df['cycle'],df[s])
         axes[i].set_ylabel('engine ' + str(e_id))
         axes[i].set_xlabel('cycle')
         #axes[i].set_title('engine ' + str(e_id), loc='right')

      #plt.tight_layout()
      plt.subplots_adjust(wspace=0, hspace=0)
      plt.savefig(f'/home/greystone/StorageWall/data_debug/Plot_debug/plot_time_series_{s}.png')
      sns.reset_orig()

   def gen_sequence(self, data, seq_length, seq_cols):
      data_matrix = data[seq_cols].values
      num_elements = data_matrix.shape[0]
      for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
         yield data_matrix[start:stop, :]

   def gen_labels(self, data, seq_length, label):
      data_matrix = data[label].values
      num_elements = data_matrix.shape[0]
      return data_matrix[seq_length:num_elements, :]

   def prepare_sequence_data(self, data, sequence_length, sequence_cols_X, sequence_cols_Y):

      #seq_gen = list(self.gen_sequence(data, sequence_length, sequence_cols_X))
      seq_gen = (list(self.gen_sequence(data[data['ID'] == id], sequence_length, sequence_cols_X)) for id in data['ID'].unique())
      seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
      #seq_array = seq_array.reshape(len(data)-sequence_length, sequence_length, len(sequence_cols_X))
      print(seq_array.shape)
      #print(seq_array)

      label_gen = [self.gen_labels(data[data['ID']==id], sequence_length, sequence_cols_Y) for id in data['ID'].unique()]
      label_array = np.concatenate(label_gen).astype(np.float32)
      
      # label_gen = [self.gen_labels(data, sequence_length, sequence_cols_Y)]
      # label_array = np.concatenate(label_gen).astype(np.float32)
      print(label_array.shape)
      #print(label_array)

      return seq_array, label_array

   def prepare_data_for_train(self, X_train_raw, y_train_raw):
      X_train = np.empty((len(X_train_raw), len(X_train_raw[1])*len(X_train_raw[2])))
      y_train = np.empty(len(X_train_raw))
      
      for i in range(len(X_train_raw)):
         X_sample_raw_data = (np.array(X_train_raw[i]).flatten()).reshape(-1)
         X_train[i] = X_sample_raw_data
         y_sample_raw_data = (np.array(y_train_raw[i]).flatten()).reshape(-1)
         y_train[i] = y_sample_raw_data
      
      print(X_train.shape)
      print(y_train.shape)
      return X_train, y_train
   
   def bin_classify(self, model, X_train, y_train, X_test, clf, params=None, score=None):

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
      
      grid_search = model_selection.GridSearchCV(estimator=clf, param_grid=params, cv=5, scoring=score, n_jobs=-1)

      grid_search.fit(X_train, y_train)
      joblib.dump(grid_search, f'/home/greystone/StorageWall/model_template/PredictMaintenance/{model}.joblib')
      y_pred = grid_search.predict(X_test)

      if hasattr(grid_search, 'predict_proba'):   
         y_score = grid_search.predict_proba(X_test)[:,1]
      elif hasattr(grid_search, 'decision_function'):
         y_score = grid_search.decision_function(X_test)
      else:
         y_score = y_pred
         
      predictions = {'y_pred' : y_pred, 'y_score' : y_score}
      df_predictions = pd.DataFrame.from_dict(predictions)

      return grid_search.best_estimator_, df_predictions
   
   def bin_class_metrics(self, model, y_test, y_pred, y_score, print_out=True, plot_out=True):
         
      """Calculate main binary classifcation metrics, plot AUC ROC and Precision-Recall curves.
      Args:
         model (str): The model name identifier
         y_test (series): Contains the test label values
         y_pred (series): Contains the predicted values
         y_score (series): Contains the predicted scores
         print_out (bool): Print the classification metrics and thresholds values
         plot_out (bool): Plot AUC ROC, Precision-Recall, and Threshold curves
      Returns:
         dataframe: The combined metrics in single dataframe
         dataframe: ROC thresholds
         dataframe: Precision-Recall thresholds
         Plot: AUC ROC
         plot: Precision-Recall
         plot: Precision-Recall threshold; also show the number of engines predicted for maintenace per period (queue).
         plot: TPR-FPR threshold
      """
      
      binclass_metrics = {
                        'Accuracy' : metrics.accuracy_score(y_test, y_pred),
                        'Precision' : metrics.precision_score(y_test, y_pred),
                        'Recall' : metrics.recall_score(y_test, y_pred),
                        'F1 Score' : metrics.f1_score(y_test, y_pred),
                        'ROC AUC' : metrics.roc_auc_score(y_test, y_score)
                        }

      df_metrics = pd.DataFrame.from_dict(binclass_metrics, orient='index')
      df_metrics.columns = [model]  


      fpr, tpr, thresh_roc = metrics.roc_curve(y_test, y_score)
      
      roc_auc = metrics.auc(fpr, tpr)

      engines_roc = []  
      for thr in thresh_roc:  
         engines_roc.append((y_score >= thr).mean())

      engines_roc = np.array(engines_roc)

      roc_thresh = {
                     'Threshold' : thresh_roc,
                     'TPR' : tpr,
                     'FPR' : fpr,
                     'Que' : engines_roc
                  }
      
      df_roc_thresh = pd.DataFrame.from_dict(roc_thresh)
      
      #calculate other classification metrics: TP, FP, TN, FN, TNR, FNR
      #from ground truth file, positive class = 25 => TP + FN = 25
      #from ground truth file, negative class = 75 => TN + FP = 75
      
      df_roc_thresh['TP'] = (25*df_roc_thresh.TPR).astype(int)
      df_roc_thresh['FP'] = (25 - (25*df_roc_thresh.TPR)).astype(int)
      df_roc_thresh['TN'] = (75*(1 - df_roc_thresh.FPR)).astype(int)
      df_roc_thresh['FN'] = (75 - (75*(1 - df_roc_thresh.FPR))).astype(int)
      
      df_roc_thresh['TNR'] = df_roc_thresh['TN']/(df_roc_thresh['TN'] + df_roc_thresh['FN'])
      df_roc_thresh['FNR'] = df_roc_thresh['TN']/(df_roc_thresh['TN'] + df_roc_thresh['FP'])
      
      df_roc_thresh['Model'] = model

      

      precision, recall, thresh_prc = metrics.precision_recall_curve(y_test, y_score)

      thresh_prc = np.append(thresh_prc,1)

      engines_prc = []  
      for thr in thresh_prc:  
         engines_prc.append((y_score >= thr).mean())

      engines_prc = np.array(engines_prc)

      prc_thresh = {
                     'Threshold' : thresh_prc,
                     'Precision' : precision,
                     'Recall' : recall,
                     'Que' : engines_prc
                  }

      df_prc_thresh = pd.DataFrame.from_dict(prc_thresh)

      if print_out:
         print('-----------------------------------------------------------')
         print(model, '\n')
         print('Confusion Matrix:')
         print(metrics.confusion_matrix(y_test, y_pred))
         print('\nClassification Report:')
         print(metrics.classification_report(y_test, y_pred))
         print('\nMetrics:')
         print(df_metrics)

         print('\nROC Thresholds:\n')
         print(df_roc_thresh[['Threshold', 'TP', 'FP', 'TN', 'FN', 'TPR', 'FPR', 'TNR','FNR', 'Que']])

         print('\nPrecision-Recall Thresholds:\n')
         print(df_prc_thresh[['Threshold', 'Precision', 'Recall', 'Que']])

      if plot_out:
         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False )
         fig.set_size_inches(10,10)

         ax1.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f'% roc_auc)
         ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
         ax1.set_xlim([-0.05, 1.0])
         ax1.set_ylim([0.0, 1.05])
         ax1.set_xlabel('False Positive Rate')
         ax1.set_ylabel('True Positive Rate')
         ax1.legend(loc="lower right", fontsize='small')

         ax2.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
         ax2.set_xlim([0.0, 1.0])
         ax2.set_ylim([0.0, 1.05])
         ax2.set_xlabel('Recall')
         ax2.set_ylabel('Precision')
         ax2.legend(loc="lower left", fontsize='small')

         ax3.plot(thresh_roc, fpr, color='red', lw=2, label='FPR')  
         ax3.plot(thresh_roc, tpr, color='green',label='TPR') 
         ax3.plot(thresh_roc, engines_roc, color='blue',label='Engines') 
         ax3.set_ylim([0.0, 1.05])
         ax3.set_xlabel('Threshold')  
         ax3.set_ylabel('%')
         ax3.legend(loc='upper right', fontsize='small')

         ax4.plot(thresh_prc, precision, color='red', lw=2, label='Precision')  
         ax4.plot(thresh_prc, recall, color='green',label='Recall') 
         ax4.plot(thresh_prc, engines_prc, color='blue',label='Engines') 
         ax4.set_ylim([0.0, 1.05])
         ax4.set_xlabel('Threshold')  
         ax4.set_ylabel('%')
         ax4.legend(loc='lower left', fontsize='small')

      return  df_metrics, df_roc_thresh, df_prc_thresh

   def run(self):
      #df_train = pd.read_excel('/home/greystone/StorageWall/data_debug/train.xlsx',header = None)
      df_train = pd.read_csv('/home/greystone/StorageWall/data_debug/augment_data.csv',header = None)
      df_train = pd.DataFrame(df_train)
      df_train = df_train.astype('float64')
      #df_train.drop([0], axis=1, inplace=True)

      features_orig = self.features
      df_train.columns = features_orig
      print(df_train.head())

      print(df_train['need_maintenance'].value_counts())
      print('Negative sample train = {}%'.format(df_train['need_maintenance'].value_counts()[0]/df_train['need_maintenance'].count() * 100))
      print('Positive sample train = {}%'.format(df_train['need_maintenance'].value_counts()[1]/df_train['need_maintenance'].count() * 100))

      df_test = pd.read_excel('/home/greystone/StorageWall/data_debug/test.xlsx',header = None)
      df_test = pd.DataFrame(df_test)
      df_test = df_test.astype('float64')
      #df_test.drop([0], axis=1, inplace=True)
      df_test.columns = features_orig
      print(df_test.head())

      df_out_train, features_extr = self.add_features(df_train, 3)
      print(df_out_train.head())
      
      df_out_test, features_extr = self.add_features(df_test, 3)
      print(df_out_test.head())

      np.savetxt('/home/greystone/StorageWall/data_debug/df_out_train.csv', df_out_train, delimiter=',')
      np.savetxt('/home/greystone/StorageWall/data_debug/df_out_test.csv', df_out_test, delimiter=',')
      
      
      y_test = df_test['need_maintenance']

      df_out_train[self.sequence_cols_X].std().plot(kind='bar', figsize=(15,15), title="Features Standard Deviation")
      plt.savefig('/home/greystone/StorageWall/data_debug/Plot_debug/plot_std.png')
      
      df_out_train[self.sequence_cols_X].std().plot(kind='bar', figsize=(15,15), logy=True,title="Features Standard Deviation (log)")
      plt.savefig('/home/greystone/StorageWall/data_debug/Plot_debug/plot_std_log.png')
      
      #df_out_train[self.sequence_cols_X].corrwith(df_tr_lbl.ttf).sort_values(ascending=False)
      df_out_train[self.sequence_cols_X].corrwith(df_out_train.need_maintenance).plot(kind='bar', figsize=(15,15), title="Correlation With label need_maintenance", fontsize=16)
      plt.savefig('/home/greystone/StorageWall/data_debug/Plot_debug/correlation_with_Y.png')
      
      correl_featurs_lbl = self.sequence_cols_X + ['need_maintenance']
      
      cm = np.corrcoef(df_out_train[correl_featurs_lbl].values.T)
      sns.set(font_scale=1.0)
      fig = plt.figure(figsize=(15, 15))
      hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 16}, yticklabels=correl_featurs_lbl, xticklabels=correl_featurs_lbl)
      hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), size=16)
      hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), size=16)
      plt.title('Features Correlation Heatmap')
      plt.savefig('/home/greystone/StorageWall/data_debug/Plot_debug/heat_map.png')
      sns.reset_orig()
      
      for item in self.sequence_cols_X:
         self.explore_col(df_out_train, item, 10)
         self.plot_time_series(df_out_train, item)
      
      sequence_cols_X = self.sequence_cols_X
      sequence_cols_Y = self.sequence_cols_Y
      
      # df_out_train_X = df_out_train[sequence_cols_X]
      # df_out_train_Y = df_out_train[sequence_cols_Y]
      
      # df_out_test_X = df_out_test[sequence_cols_X]
      # df_out_test_Y = df_out_test[sequence_cols_Y]
      
      sequence_length = 6
      X_train_raw, y_train_raw = self.prepare_sequence_data(df_out_train, sequence_length, sequence_cols_X, sequence_cols_Y)
      X_test_raw, y_test_raw = self.prepare_sequence_data(df_out_test, sequence_length, sequence_cols_X, sequence_cols_Y)
      
      print('X_train_raw shape: ', X_train_raw.shape)
      print('y_train_raw shape:', y_train_raw.shape)
      print('X_test_raw shape: ', X_test_raw.shape)
      print('y_test_raw shape:', y_test_raw.shape)
      
      X_train, y_train = self.prepare_data_for_train(X_train_raw, y_train_raw)
      X_test, y_test = self.prepare_data_for_train(X_test_raw, y_test_raw)
      
      
      
      df_out_train_X, df_out_test_X, df_out_train_Y   , df_out_test_Y = train_test_split(X_train,y_train, random_state=0)
      
      
      
      print('df_out_train_X shape: ',df_out_train_X.shape)
      print('df_out_train_Y shape:', df_out_train_Y.shape)
      print('df_out_test_X shape: ',df_out_test_X.shape)
      print('df_out_test_Y shape:', df_out_test_Y.shape)
      
      
      
      
      # __In model names:__  
      # 
      # __B__ stands for applying the model on the original features set, __B__efore feature extraction  
      # __A__ stands for applying the model on the original + extracted features set, __A__fter feature extraction  
      
      # ## Logistic Regression
      model = 'Logistic_Regression_B'
      clf_lgrb = LogisticRegression(random_state=123)
      gs_params = {'C': [.01, 0.1, 1.0, 10], 'solver': ['liblinear', 'lbfgs']}
      gs_score = 'roc_auc'
      clf_lgrb, pred_lgrb = self.bin_classify(model, df_out_train_X, df_out_train_Y, df_out_test_X, clf_lgrb, params=gs_params, score=gs_score)
      print('\nBest Parameters:\n',clf_lgrb)
      metrics_lgrb, roc_lgrb, prc_lgrb = self.bin_class_metrics(model, df_out_test_Y, pred_lgrb.y_pred, pred_lgrb.y_score, print_out=True, plot_out=True)


      model = 'Logistic_Regression_A'
      clf_lgra = LogisticRegression(random_state=123)
      gs_params = {'C': [.01, 0.1, 1.0, 10], 'solver': ['liblinear', 'lbfgs']}
      gs_score = 'roc_auc'

      clf_lgra, pred_lgra = self.bin_classify(model, df_out_train_X, df_out_train_Y, df_out_test_X, clf_lgra, params=gs_params, score=gs_score)
      print('\nBest Parameters:\n',clf_lgra)

      metrics_lgra, roc_lgra, prc_lgra = self.bin_class_metrics(model, df_out_test_Y, pred_lgra.y_pred, pred_lgra.y_score, print_out=True, plot_out=True)
      
      metrics_lgr = pd.concat([metrics_lgrb, metrics_lgra], axis=1)
      print(metrics_lgr)

      ### Support vector machine

      model = 'SVC_B'
      #clf_svcb = SGDClassifier(loss='hinge', penalty='l2')
      clf_svcb = SVC(kernel='rbf', random_state=123)
      gs_params = {'C': [0.01, 0.1, 1, 10, 100, 1000],'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'kernel': ['rbf']}
      gs_score = 'roc_auc'
      
      clf_svcb, pred_svcb = self.bin_classify(model, df_out_train_X, df_out_train_Y, df_out_test_X, clf_svcb, params=gs_params, score=gs_score)
      print('\nBest Parameters:\n',clf_svcb)
      metrics_svcb, roc_svcb, prc_svcb = self.bin_class_metrics(model, df_out_test_Y, pred_svcb.y_pred, pred_svcb.y_score, print_out=True, plot_out=True)

      model = 'SVC_A'
      #clf_svca = SGDClassifier(loss='hinge', penalty='l2')
      clf_svca = SVC(kernel='rbf', random_state=123)
      gs_params = {'C': [0.01, 0.1, 1, 10, 100, 1000],'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'kernel': ['rbf']}
      gs_score = 'roc_auc'
      clf_svca, pred_svca = self.bin_classify(model, df_out_train_X, df_out_train_Y, df_out_test_X, clf_svcb, params=gs_params, score=gs_score)
      print('\nBest Parameters:\n',clf_svca)
      
      metrics_svca, roc_svca, prc_svca = self.bin_class_metrics(model, df_out_test_Y, pred_svca.y_pred, pred_svca.y_score, print_out=True, plot_out=True)
      
      metrics_svc = pd.concat([metrics_svcb, metrics_svca], axis=1)
      print(metrics_svc.sort_index())
      
      ## SVC Linear

      model = 'SVC_Linear_B'
      clf_svlb = LinearSVC(random_state=123)
      gs_params = {'C': [.001, .01 ,.1 ,1.0]}
      gs_score = 'roc_auc'

      clf_svlb, pred_svlb = self.bin_classify(model, df_out_train_X, df_out_train_Y, df_out_test_X, clf_svlb, params=gs_params, score=gs_score)
      print('\nBest Parameters:\n',clf_svlb)

      metrics_svlb, roc_svlb, prc_svlb = self.bin_class_metrics(model, df_out_test_Y, pred_svlb.y_pred, pred_svlb.y_score, print_out=True, plot_out=True)

      model = 'SVC_Linear_A'
      clf_svla = LinearSVC(random_state=123)
      gs_params = {'C': [.001, .01 ,.1, 1.0 ]}
      gs_score = 'roc_auc'

      clf_svla, pred_svla = self.bin_classify(model, df_out_train_X, df_out_train_Y, df_out_test_X, clf_svla, params=gs_params, score=gs_score)
      print('\nBest Parameters:\n',clf_svla)

      metrics_svla, roc_svla, prc_svla = self.bin_class_metrics(model, df_out_test_Y, pred_svla.y_pred, pred_svla.y_score, print_out=True, plot_out=True)

      metrics_svl = pd.concat([metrics_svlb, metrics_svla], axis=1)
      print(metrics_svl.sort_index())
      
      ## Gaussian Naive Bayes
      model = 'Gaussian_NB_B'
      clf_gnbb = GaussianNB()
      gs_params = {} 
      gs_score = 'roc_auc'

      clf_gnbb, pred_gnbb = self.bin_classify(model, df_out_train_X, df_out_train_Y, df_out_test_X, clf_gnbb, params=gs_params, score=gs_score)
      print('\nBest Parameters:\n',clf_gnbb)

      metrics_gnbb, roc_gnbb, prc_gnbb = self.bin_class_metrics(model, df_out_test_Y, pred_gnbb.y_pred, pred_gnbb.y_score, print_out=True, plot_out=True)

      model = 'Gaussian_NB_A'
      clf_gnba = GaussianNB()
      gs_params = {} 
      gs_score = 'roc_auc'

      clf_gnba, pred_gnba = self.bin_classify(model, df_out_train_X, df_out_train_Y, df_out_test_X, clf_gnba, params=gs_params, score=gs_score) 
      print('\nBest Parameters:\n',clf_gnba)

      metrics_gnba, roc_gnba, prc_gnba = self.bin_class_metrics(model, df_out_test_Y, pred_gnba.y_pred, pred_gnba.y_score, print_out=True, plot_out=True)

      metrics_gnb = pd.concat([metrics_gnbb, metrics_gnba], axis=1)
      print(metrics_gnb.sort_index())

      ### Random forest
      model = 'Random_Forest_B'
      clf_rfcb = RandomForestClassifier(n_estimators=50, random_state=123)
      gs_params = {'max_depth': [4, 5, 6, 7, 8], 'criterion': ['gini', 'entropy']}
      gs_score = 'roc_auc'

      clf_rfcb, pred_rfcb = self.bin_classify(model, df_out_train_X, df_out_train_Y, df_out_test_X, clf_rfcb, params=gs_params, score=gs_score)
      print('\nBest Parameters:\n',clf_rfcb)

      metrics_rfcb, roc_rfcb, prc_rfcb = self.bin_class_metrics(model, df_out_test_Y, pred_rfcb.y_pred, pred_rfcb.y_score, print_out=True, plot_out=True)

      model = 'Random_Forest_A'
      clf_rfca = RandomForestClassifier(n_estimators=50, random_state=123)
      gs_params = {'max_depth': [4, 5, 6, 7, 8], 'criterion': ['gini', 'entropy']}
      gs_score = 'roc_auc'

      clf_rfca, pred_rfca = self.bin_classify(model, df_out_train_X, df_out_train_Y, df_out_test_X, clf_rfca, params=gs_params, score=gs_score)
      print('\nBest Parameters:\n',clf_rfca)

      metrics_rfca, roc_rfca, prc_rfca = self.bin_class_metrics(model, df_out_test_Y, pred_rfca.y_pred, pred_rfca.y_score, print_out=True, plot_out=True)

      metrics_rfc = pd.concat([metrics_rfcb, metrics_rfca], axis=1)
      print(metrics_rfc)

      ### KNeighbors
      model = 'KNN_B'
      clf_knnb = KNeighborsClassifier(n_jobs=-1)
      gs_params = {'n_neighbors': [9, 10, 11, 12, 13]}
      gs_score = 'roc_auc'

      clf_knnb, pred_knnb = self.bin_classify(model, df_out_train_X, df_out_train_Y, df_out_test_X, clf_knnb, params=gs_params, score=gs_score)
      print('\nBest Parameters:\n',clf_knnb)

      metrics_knnb, roc_knnb, prc_knnb = self.bin_class_metrics(model, df_out_test_Y, pred_knnb.y_pred, pred_knnb.y_score, print_out=True, plot_out=True)

      model = 'KNN_A'
      clf_knna = KNeighborsClassifier(n_jobs=-1)
      gs_params = {'n_neighbors': [9 , 10, 11, 12, 13]}
      gs_score = 'roc_auc'

      clf_knna, pred_knna = self.bin_classify(model, df_out_train_X, df_out_train_Y, df_out_test_X, clf_knna, params=gs_params, score=gs_score)
      print('\nBest Parameters:\n',clf_knna)

      metrics_knna, roc_knna, prc_knna = self.bin_class_metrics(model, df_out_test_Y, pred_knna.y_pred, pred_knna.y_score, print_out=True, plot_out=True)

      # Compare KNN Before and After FE
      metrics_knn = pd.concat([metrics_knnb, metrics_knna], axis=1)
      print(metrics_knn.sort_index())

      metrics_bn = pd.concat([metrics_lgr, metrics_rfc, metrics_svc, metrics_svl, metrics_knn, metrics_gnb], axis=1)
      print(metrics_bn)


      fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False )
      fig.set_size_inches(10,5)

      ax1.plot(roc_gnbb.FPR, roc_gnbb.TPR, color='yellow', lw=1, label= metrics_gnbb.columns.values.tolist()[0] + ':  %.3f' % metrics_gnbb.at['ROC AUC', metrics_gnbb.columns.values.tolist()[0]])
      ax1.plot(roc_rfca.FPR, roc_rfca.TPR, color='green', lw=1, label= metrics_rfca.columns.values.tolist()[0] + ':  %.3f' % metrics_rfca.at['ROC AUC', metrics_rfca.columns.values.tolist()[0]])
      ax1.plot(roc_lgra.FPR, roc_lgra.TPR, color='blue', lw=1, label= metrics_lgra.columns.values.tolist()[0] + ':  %.3f' % metrics_lgra.at['ROC AUC', metrics_lgra.columns.values.tolist()[0]])
      ax1.plot(roc_svla.FPR, roc_svla.TPR, color='brown', lw=1, label= metrics_svla.columns.values.tolist()[0] + ':  %.3f' % metrics_svla.at['ROC AUC', metrics_svla.columns.values.tolist()[0]])
      ax1.plot(roc_svlb.FPR, roc_svlb.TPR, color='sandybrown', lw=1, label= metrics_svlb.columns.values.tolist()[0] + ':  %.3f' % metrics_svlb.at['ROC AUC', metrics_svlb.columns.values.tolist()[0]])
      ax1.plot(roc_knna.FPR, roc_knna.TPR, color='darkmagenta', lw=1, label= metrics_knna.columns.values.tolist()[0] + ':  %.3f' % metrics_knna.at['ROC AUC', metrics_knna.columns.values.tolist()[0]])
      ax1.plot(roc_svca.FPR, roc_svca.TPR, color='red', lw=1, label= metrics_svca.columns.values.tolist()[0] + ':  %.3f' % metrics_svca.at['ROC AUC', metrics_svca.columns.values.tolist()[0]])
      ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
      ax1.set_xlim([-0.05, 1.0])
      ax1.set_ylim([0.0, 1.05])
      ax1.set_xlabel('False Positive Rate')
      ax1.set_ylabel('True Positive Rate')
      ax1.legend(loc="lower right", fontsize='small')
      ax1.set_title('AUC ROC')

      ax2.plot(prc_gnbb.Recall, prc_gnbb.Precision, color='yellow', lw=1, label= metrics_gnbb.columns.values.tolist()[0])
      ax2.plot(prc_rfca.Recall, prc_rfca.Precision, color='green', lw=1, label= metrics_rfca.columns.values.tolist()[0])
      ax2.plot(prc_lgra.Recall, prc_lgra.Precision, color='blue', lw=1, label= metrics_lgra.columns.values.tolist()[0])
      ax2.plot(prc_svla.Recall, prc_svla.Precision, color='brown', lw=1, label= metrics_svla.columns.values.tolist()[0])
      ax2.plot(prc_svlb.Recall, prc_svlb.Precision, color='sandybrown', lw=1, label= metrics_svlb.columns.values.tolist()[0])
      ax2.plot(prc_knna.Recall, prc_knna.Precision, color='darkmagenta', lw=1, label= metrics_knna.columns.values.tolist()[0])
      ax2.plot(prc_svca.Recall, prc_svca.Precision, color='red', lw=1, label= metrics_svca.columns.values.tolist()[0])
      ax2.set_xlim([0.0, 1.0])
      ax2.set_ylim([0.0, 1.05])
      ax2.set_xlabel('Recall')
      ax2.set_ylabel('Precision')
      ax2.legend(loc="lower left", fontsize='small')
      ax2.set_title('Precision Recall Curve')
      plt.savefig('/home/greystone/StorageWall/data_debug/Plot_debug/AUC_ROC_precision_recall_curves_for_best_models.png')


if __name__ == '__main__':
   model = Train_Predictive_Maintenance()
   model.run()
   
   # df_train = pd.read_excel('/home/greystone/StorageWall/data_debug/train.xlsx',header = None)
   # df_train = pd.DataFrame(df_train)
   # df_train = df_train.astype('float64')
   # df_train.columns = ['cycle', 'retry_clamp', 'retry_release', 'life_time', 'drop_decive', 'clamp_elapse_time', 'release_elapse_time', 'need_maintenance']
   
   # print(df_train.head())
   
   # print(df_train['release_elapse_time'][2])
   # print(df_train.shape)

   # data_col = []
   
   # sequence_cols_X = ['cycle', 'retry_clamp', 'retry_release', 'life_time', 'drop_decive', 'clamp_elapse_time', 'release_elapse_time']
   # sequence_cols_Y = ['need_maintenance']
   
   # def gen_sequence(id_df, seq_length, seq_cols):
   #    data_matrix = id_df[seq_cols].values
   #    num_elements = data_matrix.shape[0]

   #    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
   #       yield data_matrix[start:stop, :]
   
   # def gen_labels(id_df, seq_length, label):
   #    data_matrix = id_df[label].values
   #    num_elements = data_matrix.shape[0]
   #    return data_matrix[seq_length:num_elements, :]
   
   # sequence_length = 6
   
   # def prepare_sequence_data(data, sequence_length, sequence_cols_X, sequence_cols_Y):

   #    seq_gen = list(gen_sequence(data, sequence_length, sequence_cols_X))
   #    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
   #    seq_array = seq_array.reshape(20, 6, 7)
   #    print(seq_array.shape)
   #    print(seq_array)
   
   #    label_gen = [gen_labels(data, sequence_length, sequence_cols_Y)]
   #    label_array = np.concatenate(label_gen).astype(np.float32)
   #    print(label_array.shape)
   #    print(label_array)
      
   #    return seq_array, label_array
   
   