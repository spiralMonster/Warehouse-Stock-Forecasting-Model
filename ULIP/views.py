from django.shortcuts import render
from django.http import JsonResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from .Models.prophet_model import ProphetModel
from .Models.lstm_cnn_hybrid_model import LSTMAndCNN4StockForecasting
import joblib
import pandas as pd
import numpy as np
import json

def get_stock_forecasting(request,*args,**kwargs):
    if request.method=='POST':
        try:
            data=json.loads(request.body.decode("utf-8"))
            data=pd.DataFrame(data)

        except Exception as e:
            print(e)

        with open('encoder.pkl','rb') as file:
            encoder=joblib.load(file)

        file.close()
        data_X=encoder.transform(data)

        features = [
            'Stock Level Thresholds',
            'Seasonality',
            'Market Changes',
            'Product Type',
            'Lead time (in days)',
            'Supplier Reliabilty',
            'Stock Handing Efficiency',
            'Product Costs(In Rs.)',
            'Maximum discount offered (in percentage)',
            'Products Expiry (in months)',
            'Backorders',
            'Bulk orders (By customers)'
        ]
        dates = [
            '2023-12-01',
            '2024-01-01',
            '2024-02-01',
            '2024-03-01',
            '2024-04-01',
            '2024-05-01',
            '2024-06-01',
            '2024-07-01',
            '2024-08-01',
            '2024-09-01',
            '2024-10-01',
            '2024-11-01',
            '2024-12-01'
        ]
        months = ['Dec-2023',
                  'Jan-2024,',
                  'Feb-2024',
                  'Mar-2024',
                  'Apr-2024',
                  'May-2024',
                  'Jun-2024',
                  'Jul-2024',
                  'Aug-2024',
                  'Sep-2024',
                  'Oct-2024',
                  'Nov-2024',
                  'Dec-2024'
        ]

        data_by_month = []
        for i in range(len(data)):
            exp = []
            for j in range(13):
                cols = ['Product Name', 'Product Category']
                cols.append(f'Stocks Required/{dates[j]}')
                for feat in features:
                    cols.append(feat + f'-{months[j]}')
                inst = list(data_X.loc[i, cols])
                exp.append(inst)
            data_by_month.append(exp)
        data_by_month = np.array(data_by_month)
        lstm_cnn_hybrid_data=data_by_month

        columns_for_prophet_model=[f'Stocks Required/{date}' for date in dates]
        prophet_model_data=data[columns_for_prophet_model]
        prophet_model_data=np.array(prophet_model_data)
        with CustomObjectScope({
            'ProphetModel':ProphetModel,
            'LSTMAndCNN4StockForecasting':LSTMAndCNN4StockForecasting
        }):
            model=load_model('ulip_model_stock_forecasting.h5')

        predictions=model.predict([prophet_model_data,lstm_cnn_hybrid_data])
        pred=tf.round(predictions)
        pred=tf.cast(pred,tf.int32)
        stock_forecasting={}
        for product,stock in zip(data['Product Name'],pred):
            stock_forecasting[product]=stock

        try:
            return JsonResponse(stock_forecasting,status=200)

        except Exception as e:
            return JsonResponse({'error':str(e)},status=500)

    else:
        print("The method is not a post method!!!")



