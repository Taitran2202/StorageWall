
import numpy as np
import pandas as pd


def add_gaussian_noise(time_series_ID, time_series_cycle, time_series_input, time_series_output, times_augment, mean, stddev):
   """
   Adds Gaussian noise to a time series.

   Options:
   time_series_input (array-like): A time series to which noise is added.
   time_series_output (array-like): A time series to which noise is added.
   mean (float): The average value of the noise. Default is 0.0.
   stddev (float): Standard deviation of noise. Default is 1.0.

   Returns:
   noisy_series (np.array): Time series with added noise.
   """
   augmented_ID = []
   augmented_cycle = []
   augmented_input = []
   augmented_output = []
   
   len_aug_per_id = 0
   flag = 0
   
   for i in range(len(time_series_input)):
      augmented_ID.append(time_series_ID[i])
      if time_series_ID[i] == 1:
         augmented_cycle.append(time_series_cycle[i] + times_augment*i)
      else:
         if time_series_ID[i] != time_series_ID[i-1]:
            degree_count = len_aug_per_id
            flag = 1
         augmented_cycle.append(time_series_cycle[i] + (times_augment*i) - degree_count)
      augmented_input.append(time_series_input[i])
      augmented_output.append(time_series_output[i])
      for count in range(times_augment):
         len_aug_per_id += 1
         noise = np.random.normal(mean, stddev, size=(len(time_series_input[1])))
         augmented_ID.append(time_series_ID[i])
         if flag == 1:
            augmented_cycle.append(time_series_cycle[i] + (times_augment * i) + count + 1 - degree_count)
         else:
            augmented_cycle.append(time_series_cycle[i] + (times_augment * i) + count + 1 )
         augmented_input.append(time_series_input[i] + (time_series_input[i] * noise))
         augmented_output.append(time_series_output[i])

   augmented_ID = np.array(augmented_ID)
   augmented_ID = augmented_ID.reshape(-1, 1)
   augmented_cycle = np.array(augmented_cycle)
   augmented_cycle = augmented_cycle.reshape(-1, 1)
   augmented_input = np.array(augmented_input)
   augmented_output = np.array(augmented_output)
   augmented_output = augmented_output.reshape(-1, 1)
   return augmented_ID, augmented_cycle, augmented_input, augmented_output

time_series_data = pd.read_excel('/home/greystone/StorageWall/data_debug/train.xlsx',skiprows=0)
time_series_data = np.array(time_series_data)
time_series_data = time_series_data.astype('float64')

time_series_ID = time_series_data[:,0]
time_series_cycle = time_series_data[:,1]
time_series_output = time_series_data[:,-1]
time_series_input = time_series_data[:,2:-1]

times_augment = 5

augmented_ID, augmented_cycle, augmented_input, augmented_output = add_gaussian_noise(time_series_ID, time_series_cycle, time_series_input, time_series_output, times_augment, mean=0.0, stddev=0.1)

print(augmented_ID.shape)
print(augmented_cycle.shape)
print(augmented_input.shape)
print(augmented_output.shape)

augmented_time_series_data = np.hstack((augmented_ID, augmented_cycle, augmented_input, augmented_output))
# np.concatenate((augmented_input, augmented_output), axis=1)

np.savetxt('/home/greystone/StorageWall/data_debug/augment_data.csv', augmented_time_series_data, delimiter=',')
print(augmented_time_series_data)