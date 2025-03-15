{\rtf1\ansi\ansicpg1252\cocoartf2821
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red15\green112\blue1;\red245\green245\blue245;\red0\green0\blue0;
\red157\green0\blue210;\red144\green1\blue18;\red19\green85\blue52;\red0\green0\blue255;\red31\green99\blue128;
\red101\green76\blue29;}
{\*\expandedcolortbl;;\cssrgb\c0\c50196\c0;\cssrgb\c96863\c96863\c96863;\cssrgb\c0\c0\c0;
\cssrgb\c68627\c0\c85882;\cssrgb\c63922\c8235\c8235;\cssrgb\c6667\c40000\c26667;\cssrgb\c0\c0\c100000;\cssrgb\c14510\c46275\c57647;
\cssrgb\c47451\c36863\c14902;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 #### Final working code LSTM + CNN\cf0 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 import\cf0 \strokec4  numpy \cf5 \strokec5 as\cf0 \strokec4  np\cb1 \
\cf5 \cb3 \strokec5 from\cf0 \strokec4  tensorflow.keras.models \cf5 \strokec5 import\cf0 \strokec4  Sequential\cb1 \
\cf5 \cb3 \strokec5 from\cf0 \strokec4  tensorflow.keras.layers \cf5 \strokec5 import\cf0 \strokec4  Conv1D, MaxPooling1D, LSTM, Dense\cb1 \
\cf5 \cb3 \strokec5 from\cf0 \strokec4  tensorflow.keras.layers \cf5 \strokec5 import\cf0 \strokec4  Dense, Dropout\cb1 \
\cf5 \cb3 \strokec5 import\cf0 \strokec4  pandas \cf5 \strokec5 as\cf0 \strokec4  pd\cb1 \
\cf5 \cb3 \strokec5 from\cf0 \strokec4  matplotlib \cf5 \strokec5 import\cf0 \strokec4  pyplot \cf5 \strokec5 as\cf0 \strokec4  plt\cb1 \
\cf5 \cb3 \strokec5 from\cf0 \strokec4  sklearn.preprocessing \cf5 \strokec5 import\cf0 \strokec4  StandardScaler\cb1 \
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 #from datetime import datetime\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 # Load the dataset\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 df = pd.read_csv(\cf6 \strokec6 "/content/sample_data/Sales_Data_Fsd_All.csv"\cf0 \strokec4 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Convert 'Date' column to datetime\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 df[\cf6 \strokec6 'Date'\cf0 \strokec4 ] = pd.to_datetime(df[\cf6 \strokec6 'Date'\cf0 \strokec4 ], format=\cf6 \strokec6 'mixed'\cf0 \strokec4 )\cb1 \
\
\cb3 df[\cf6 \strokec6 'DayOfWeek'\cf0 \strokec4 ] = df[\cf6 \strokec6 'Date'\cf0 \strokec4 ].dt.dayofweek\cb1 \
\cb3 df[\cf6 \strokec6 'Month'\cf0 \strokec4 ] = df[\cf6 \strokec6 'Date'\cf0 \strokec4 ].dt.month\cb1 \
\
\cb3 df[\cf6 \strokec6 'DaySin'\cf0 \strokec4 ] = np.sin(\cf7 \strokec7 2\cf0 \strokec4  * np.pi * df[\cf6 \strokec6 'DayOfWeek'\cf0 \strokec4 ] / \cf7 \strokec7 7\cf0 \strokec4 )\cb1 \
\cb3 df[\cf6 \strokec6 'DayCos'\cf0 \strokec4 ] = np.cos(\cf7 \strokec7 2\cf0 \strokec4  * np.pi * df[\cf6 \strokec6 'DayOfWeek'\cf0 \strokec4 ] / \cf7 \strokec7 7\cf0 \strokec4 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Feature Engineering (Example: Lagged Sales)\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 df[\cf6 \strokec6 'Lagged_Sales'\cf0 \strokec4 ] = df[\cf6 \strokec6 'Sale'\cf0 \strokec4 ].shift(\cf7 \strokec7 1\cf0 \strokec4 )\cb1 \
\cb3 df = df.dropna()  \cf2 \strokec2 # Remove rows with NaN values after shifting\cf0 \cb1 \strokec4 \
\
\cb3 df[\cf6 \strokec6 'Sales_Lag7'\cf0 \strokec4 ] = df[\cf6 \strokec6 'Sale'\cf0 \strokec4 ].shift(\cf7 \strokec7 7\cf0 \strokec4 )\cb1 \
\
\cb3 df[\cf6 \strokec6 'Sales_MA_7'\cf0 \strokec4 ] = df[\cf6 \strokec6 'Sale'\cf0 \strokec4 ].rolling(window=\cf7 \strokec7 7\cf0 \strokec4 ).mean()\cb1 \
\cb3 df = df.dropna()  \cf2 \strokec2 # Remove NaN values after calculating moving average\cf0 \cb1 \strokec4 \
\
\cb3 df[\cf6 \strokec6 'Sales_Std7'\cf0 \strokec4 ] = df[\cf6 \strokec6 'Sale'\cf0 \strokec4 ].rolling(window=\cf7 \strokec7 7\cf0 \strokec4 ).std()\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Calculate lagged differences\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 df[\cf6 \strokec6 'Sales_Diff'\cf0 \strokec4 ] = df[\cf6 \strokec6 'Sale'\cf0 \strokec4 ] - df[\cf6 \strokec6 'Sale'\cf0 \strokec4 ].shift(\cf7 \strokec7 1\cf0 \strokec4 )\cb1 \
\cb3 df = df.dropna()\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Create day of the week feature (0 = Monday, 6 = Sunday)\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 df[\cf6 \strokec6 'Day_of_Week11'\cf0 \strokec4 ] = df[\cf6 \strokec6 'Date'\cf0 \strokec4 ].dt.dayofweek  \cf2 \strokec2 # Create 'Day_of_Week11' before selecting features\cf0 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Select features for the model\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 features = [\cf6 \strokec6 'Sale'\cf0 \strokec4 , \cf6 \strokec6 'Lagged_Sales'\cf0 \strokec4 , \cf6 \strokec6 'DaySin'\cf0 \strokec4 , \cf6 \strokec6 'DayCos'\cf0 \strokec4 , \cf6 \strokec6 'Sales_Lag7'\cf0 \strokec4 , \cf6 \strokec6 'Status'\cf0 \strokec4 , \cf6 \strokec6 'EffectiveRain'\cf0 \strokec4 , \cf6 \strokec6 'Sales_MA_7'\cf0 \strokec4 , \cf6 \strokec6 'EffectiveTemperature'\cf0 \strokec4 , \cf6 \strokec6 'Sales_Diff'\cf0 \strokec4 , \cf6 \strokec6 'Sales_Std7'\cf0 \strokec4 , \cf6 \strokec6 'Day_of_Week11'\cf0 \strokec4 , \cf6 \strokec6 'PositiveEvents'\cf0 \strokec4 , \cf6 \strokec6 'NegativeEvents'\cf0 \strokec4 ]\cb1 \
\cb3 data = df[features]\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # One-hot encode day of the week to add categorical weekday effect\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 data = pd.get_dummies(data, columns=[\cf6 \strokec6 'Day_of_Week11'\cf0 \strokec4 ], drop_first=\cf8 \strokec8 True\cf0 \strokec4 ) \cf2 \strokec2 # One-hot encode after selecting features\cf0 \cb1 \strokec4 \
\
\
\
\cb3 data = data.values \cf2 \strokec2 # Convert to NumPy array after value replacement\cf0 \cb1 \strokec4 \
\
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 #Variables for training\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 cols = \cf9 \strokec9 list\cf0 \strokec4 (df)[\cf7 \strokec7 1\cf0 \strokec4 :\cf7 \strokec7 13\cf0 \strokec4 ]\cb1 \
\pard\pardeftab720\partightenfactor0
\cf10 \cb3 \strokec10 print\cf0 \strokec4 (cols)\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 #New dataframe with only training data - 5 columns\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 df_for_training = df[cols]\cb1 \
\
\
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Convert 'Status' values in df_for_training to numeric representations:\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 df_for_training[\cf6 \strokec6 'Status'\cf0 \strokec4 ].replace([\cf6 \strokec6 'OPEN'\cf0 \strokec4 , \cf6 \strokec6 'CLOSED'\cf0 \strokec4 ], [\cf7 \strokec7 0\cf0 \strokec4 , \cf7 \strokec7 1\cf0 \strokec4 ], inplace=\cf8 \strokec8 True\cf0 \strokec4 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Convert 'EffectiveRain values within the DataFrame\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 df_for_training[\cf6 \strokec6 'EffectiveRain'\cf0 \strokec4 ].replace([\cf6 \strokec6 'No'\cf0 \strokec4 , \cf6 \strokec6 'EffectiveRain'\cf0 \strokec4 ],\cb1 \
\cb3                         [\cf7 \strokec7 0\cf0 \strokec4 , \cf7 \strokec7 1\cf0 \strokec4 ], inplace=\cf8 \strokec8 True\cf0 \strokec4 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Convert 'EffectiveTemperature values within the DataFrame\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 df_for_training[\cf6 \strokec6 'EffectiveTemperature'\cf0 \strokec4 ].replace([\cf6 \strokec6 'No'\cf0 \strokec4 , \cf6 \strokec6 'HotWeather'\cf0 \strokec4 ],\cb1 \
\cb3                         [\cf7 \strokec7 0\cf0 \strokec4 , \cf7 \strokec7 1\cf0 \strokec4 ], inplace=\cf8 \strokec8 True\cf0 \strokec4 )\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Convert 'PositiveEvents values within the DataFrame\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 df_for_training[\cf6 \strokec6 'PositiveEvents'\cf0 \strokec4 ].replace([\cf6 \strokec6 'No'\cf0 \strokec4 , \cf6 \strokec6 'SalaryDay'\cf0 \strokec4 , \cf6 \strokec6 'Holiday'\cf0 \strokec4 , \cf6 \strokec6 'Promotions'\cf0 \strokec4 , \cf6 \strokec6 'Event'\cf0 \strokec4 ],\cb1 \
\cb3                         [\cf7 \strokec7 0\cf0 \strokec4 , \cf7 \strokec7 1\cf0 \strokec4 , \cf7 \strokec7 2\cf0 \strokec4 , \cf7 \strokec7 3\cf0 \strokec4 , \cf7 \strokec7 4\cf0 \strokec4 ], inplace=\cf8 \strokec8 True\cf0 \strokec4 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Convert 'NegativeEvents values within the DataFrame\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 df_for_training[\cf6 \strokec6 'NegativeEvents'\cf0 \strokec4 ].replace([\cf6 \strokec6 'No'\cf0 \strokec4 , \cf6 \strokec6 'Wednesday'\cf0 \strokec4 , \cf6 \strokec6 'Protest'\cf0 \strokec4 , \cf6 \strokec6 'PartiallyClosed'\cf0 \strokec4 ],\cb1 \
\cb3                         [\cf7 \strokec7 0\cf0 \strokec4 , \cf7 \strokec7 1\cf0 \strokec4 , \cf7 \strokec7 2\cf0 \strokec4 , \cf7 \strokec7 3\cf0 \strokec4 ], inplace=\cf8 \strokec8 True\cf0 \strokec4 )\cb1 \
\
\
\
\cb3 df_for_training = df_for_training.astype(\cf9 \strokec9 int\cf0 \strokec4 )\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 #LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized\cf0 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 # normalize the dataset\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 scaler = StandardScaler()\cb1 \
\cb3 scaler = scaler.fit(df_for_training)\cb1 \
\cb3 df_for_training_scaled = scaler.transform(df_for_training)\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 #Empty lists to be populated using formatted training data\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 trainX = []\cb1 \
\cb3 trainY = []\cb1 \
\
\cb3 n_future = \cf7 \strokec7 1\cf0 \strokec4    \cf2 \strokec2 # Number of days we want to look into the future based on the past days.\cf0 \cb1 \strokec4 \
\cb3 n_past = \cf7 \strokec7 60\cf0 \strokec4   \cf2 \strokec2 # Number of past days we want to use to predict the future.\cf0 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 for\cf0 \strokec4  i \cf8 \strokec8 in\cf0 \strokec4  \cf10 \strokec10 range\cf0 \strokec4 (n_past, \cf10 \strokec10 len\cf0 \strokec4 (df_for_training_scaled) - n_future +\cf7 \strokec7 1\cf0 \strokec4 ):\cb1 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3     trainX.append(df_for_training_scaled[i - n_past:i, \cf7 \strokec7 0\cf0 \strokec4 :df_for_training.shape[\cf7 \strokec7 1\cf0 \strokec4 ]])\cb1 \
\cb3     trainY.append(df_for_training_scaled[i + n_future - \cf7 \strokec7 1\cf0 \strokec4 :i + n_future, \cf7 \strokec7 0\cf0 \strokec4 ])\cb1 \
\
\cb3 trainX, trainY = np.array(trainX), np.array(trainY)\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf10 \cb3 \strokec10 print\cf0 \strokec4 (\cf6 \strokec6 'trainX shape == \{\}.'\cf0 \strokec4 .\cf10 \strokec10 format\cf0 \strokec4 (trainX.shape))\cb1 \
\cf10 \cb3 \strokec10 print\cf0 \strokec4 (\cf6 \strokec6 'trainY shape == \{\}.'\cf0 \strokec4 .\cf10 \strokec10 format\cf0 \strokec4 (trainY.shape))\cb1 \
\
\
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # define the Autoencoder model\cf0 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 model = Sequential()\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # LSTM + CNN\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 model.add(Conv1D(filters=\cf7 \strokec7 64\cf0 \strokec4 , kernel_size=\cf7 \strokec7 4\cf0 \strokec4 , activation=\cf6 \strokec6 'relu'\cf0 \strokec4 , input_shape=(trainX.shape[\cf7 \strokec7 1\cf0 \strokec4 ], trainX.shape[\cf7 \strokec7 2\cf0 \strokec4 ])))  \cf2 \strokec2 # CNN layer\cf0 \cb1 \strokec4 \
\cb3 model.add(MaxPooling1D(pool_size=\cf7 \strokec7 2\cf0 \strokec4 ))  \cf2 \strokec2 # Pooling layer\cf0 \cb1 \strokec4 \
\cb3 model.add(LSTM(\cf7 \strokec7 50\cf0 \strokec4 , return_sequences=\cf8 \strokec8 False\cf0 \strokec4 ))  \cf2 \strokec2 # LSTM layer\cf0 \cb1 \strokec4 \
\cb3 model.add(Dense(trainY.shape[\cf7 \strokec7 1\cf0 \strokec4 ]))\cb1 \
\cb3 model.\cf10 \strokec10 compile\cf0 \strokec4 (optimizer=\cf6 \strokec6 'adam'\cf0 \strokec4 , loss=\cf6 \strokec6 'mse'\cf0 \strokec4 )\cb1 \
\
\cb3 model.summary()\cb1 \
\
\
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # fit the model\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 history = model.fit(trainX, trainY, epochs=\cf7 \strokec7 150\cf0 \strokec4 , batch_size=\cf7 \strokec7 32\cf0 \strokec4 , validation_split=\cf7 \strokec7 0.1\cf0 \strokec4 , verbose=\cf7 \strokec7 1\cf0 \strokec4 )\cb1 \
\
\cb3 plt.plot(history.history[\cf6 \strokec6 'loss'\cf0 \strokec4 ], label=\cf6 \strokec6 'Training loss'\cf0 \strokec4 )\cb1 \
\cb3 plt.plot(history.history[\cf6 \strokec6 'val_loss'\cf0 \strokec4 ], label=\cf6 \strokec6 'Validation loss'\cf0 \strokec4 )\cb1 \
\cb3 plt.legend()\cb1 \
\
\
\cb3 n_past = \cf7 \strokec7 60\cf0 \cb1 \strokec4 \
\cb3 n_days_for_prediction=\cf7 \strokec7 60\cf0 \strokec4   \cf2 \strokec2 #let us predict past 60 days\cf0 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Get training dates from the 'Date' column of the original DataFrame\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 train_dates = df[\cf6 \strokec6 'Date'\cf0 \strokec4 ]  \cf2 \strokec2 # Assuming 'Date' column contains datetime objects\cf0 \cb1 \strokec4 \
\
\
\cb3 predict_period_dates = pd.date_range(\cf9 \strokec9 list\cf0 \strokec4 (train_dates)[-n_past], periods=n_days_for_prediction).tolist()\cb1 \
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # print(predict_period_dates)\cf0 \cb1 \strokec4 \
\
\cf2 \cb3 \strokec2 #Make prediction\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 prediction = model.predict(trainX[-n_days_for_prediction:]) \cf2 \strokec2 #shape = (n, 1) where n is the n_days_for_prediction\cf0 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 #Perform inverse transformation to rescale back to original range\cf0 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 #Since we used 5 variables for transform, the inverse expects same dimensions\cf0 \cb1 \strokec4 \
\cf2 \cb3 \strokec2 #Therefore, let us copy our values 5 times and discard them after inverse transform\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 prediction_copies = np.repeat(prediction, df_for_training.shape[\cf7 \strokec7 1\cf0 \strokec4 ], axis=\cf7 \strokec7 -1\cf0 \strokec4 )\cb1 \
\cb3 y_pred_future = scaler.inverse_transform(prediction_copies)[:,\cf7 \strokec7 0\cf0 \strokec4 ]\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Convert timestamp to date\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 forecast_dates = []\cb1 \
\pard\pardeftab720\partightenfactor0
\cf5 \cb3 \strokec5 for\cf0 \strokec4  time_i \cf8 \strokec8 in\cf0 \strokec4  predict_period_dates:\cb1 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3     forecast_dates.append(time_i.date())\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Get the last 'n_days_for_prediction' values from the 'Sale' column\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 original_sales = df[\cf6 \strokec6 'Sale'\cf0 \strokec4 ].tail(n_days_for_prediction).values\cb1 \
\
\cb3 df_forecast = pd.DataFrame(\{\cf6 \strokec6 'Date'\cf0 \strokec4 : np.array(forecast_dates),\cb1 \
\cb3                             \cf6 \strokec6 'Sale'\cf0 \strokec4 : y_pred_future.astype(\cf9 \strokec9 int\cf0 \strokec4 ),\cb1 \
\cb3                             \cf6 \strokec6 'Original'\cf0 \strokec4 : original_sales\})\cb1 \
\cb3 df_forecast[\cf6 \strokec6 'Date'\cf0 \strokec4 ]=pd.to_datetime(df_forecast[\cf6 \strokec6 'Date'\cf0 \strokec4 ])\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Calculate Mean Absolute Error (MAE)\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 mae = np.mean(np.\cf10 \strokec10 abs\cf0 \strokec4 (df_forecast[\cf6 \strokec6 'Sale'\cf0 \strokec4 ] - df_forecast[\cf6 \strokec6 'Original'\cf0 \strokec4 ]))\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Calculate Root Mean Squared Error (RMSE)\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 rmse = np.sqrt(np.mean((df_forecast[\cf6 \strokec6 'Sale'\cf0 \strokec4 ] - df_forecast[\cf6 \strokec6 'Original'\cf0 \strokec4 ])**\cf7 \strokec7 2\cf0 \strokec4 ))\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Calculate Mean Absolute Percentage Error (MAPE)\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 mape = np.mean(np.\cf10 \strokec10 abs\cf0 \strokec4 ((df_forecast[\cf6 \strokec6 'Sale'\cf0 \strokec4 ] - df_forecast[\cf6 \strokec6 'Original'\cf0 \strokec4 ]) / df_forecast[\cf6 \strokec6 'Original'\cf0 \strokec4 ])) * \cf7 \strokec7 100\cf0 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Print the results\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf10 \cb3 \strokec10 print\cf0 \strokec4 (\cf8 \strokec8 f\cf6 \strokec6 "MAE: \cf0 \strokec4 \{mae\}\cf6 \strokec6 "\cf0 \strokec4 )\cb1 \
\cf10 \cb3 \strokec10 print\cf0 \strokec4 (\cf8 \strokec8 f\cf6 \strokec6 "RMSE: \cf0 \strokec4 \{rmse\}\cf6 \strokec6 "\cf0 \strokec4 )\cb1 \
\cf10 \cb3 \strokec10 print\cf0 \strokec4 (\cf8 \strokec8 f\cf6 \strokec6 "MAPE: \cf0 \strokec4 \{mape\cf7 \strokec7 :.2f\cf0 \strokec4 \}\cf6 \strokec6 %"\cf0 \strokec4 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 original = df[[\cf6 \strokec6 'Date'\cf0 \strokec4 , \cf6 \strokec6 'Sale'\cf0 \strokec4 ]]\cb1 \
\cb3 original[\cf6 \strokec6 'Date'\cf0 \strokec4 ]=pd.to_datetime(original[\cf6 \strokec6 'Date'\cf0 \strokec4 ])\cb1 \
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # original = original.loc[original['Date'] >= '2020-5-1']\cf0 \cb1 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf10 \cb3 \strokec10 print\cf0 \strokec4 (df_forecast)\cb1 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 df_forecast.to_csv(\cf6 \strokec6 'forecast_sales_results.csv'\cf0 \strokec4 , index=\cf8 \strokec8 False\cf0 \strokec4 )\cb1 \
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 # Plotting the forecast\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 plt.figure(figsize=(\cf7 \strokec7 12\cf0 \strokec4 , \cf7 \strokec7 6\cf0 \strokec4 ))  \cf2 \strokec2 # Adjust figure size if needed\cf0 \cb1 \strokec4 \
\cb3 plt.plot(df_forecast[\cf6 \strokec6 'Date'\cf0 \strokec4 ], df_forecast[\cf6 \strokec6 'Sale'\cf0 \strokec4 ], label=\cf6 \strokec6 'Forecast'\cf0 \strokec4 )\cb1 \
\cb3 plt.plot(df_forecast[\cf6 \strokec6 'Date'\cf0 \strokec4 ], df_forecast[\cf6 \strokec6 'Original'\cf0 \strokec4 ], label=\cf6 \strokec6 'Original'\cf0 \strokec4 )\cb1 \
\cb3 plt.xlabel(\cf6 \strokec6 'Date'\cf0 \strokec4 )\cb1 \
\cb3 plt.ylabel(\cf6 \strokec6 'Sale'\cf0 \strokec4 )\cb1 \
\cb3 plt.title(\cf6 \strokec6 'Sales Forecast vs. Original Using LSTM and CNN together'\cf0 \strokec4 )\cb1 \
\cb3 plt.legend()\cb1 \
\cb3 plt.grid(\cf8 \strokec8 True\cf0 \strokec4 )\cb1 \
\cb3 plt.show()\cb1 \
}