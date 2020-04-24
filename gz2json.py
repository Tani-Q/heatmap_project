# -*- coding:utf-8 -*-
import sys
import csv
import gzip
import json
import datetime


import os

import pandas as pd 
import google.cloud.storage
from google.cloud import storage

import re

import shutil


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/tani_kyuichiro/nurve-cloud-98b3f-bb7c97f8fb03.json'


#------------------------------------ メソッド ------------------------------------------------

#GCSへのデータアップローダー    
def upload_blob(bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        # bucket_name = "your-bucket-name"
        # source_file_name = "local/path/to/file"
        # destination_blob_name = "storage-object-name"

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )


#GCSからのデータダウンロード
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )
#------------------------------------ メソッド ------------------------------------------------

def main():
    #元データダウンロード
    print('gzipファイルダウンロード中')
    bucket_name = 'heatmap-staging'
    source_blob_name = 'logging/heatmap.csv.gz'
    destination_file_name = 'data/heatmap.csv.gz'
    download_blob(bucket_name, source_blob_name, destination_file_name)
    print('gzipファイルダウンロード完了')

    #データ処理
    print('データ処理開始')
    df = pd.read_csv('./data/heatmap.csv.gz', sep=',')

    media_id_lst = df['media_id'].values.tolist()
    unique_media_id_lst = list(set(media_id_lst))

    #media_id別データ選別
    print('media_id別ファイル処理中')
    for media_id in unique_media_id_lst:    

        tmp1_dic = {}
        tmp2_dic = {}
        tmp_lst = []
       
        #データ選別
        df_tmp = df[df['media_id'] == media_id ].reset_index(drop=True)
        
        media_id = str(int(df_tmp.iloc[0]['media_id']))
        print('media_id:{}',format(media_id))
        #tmp_dic['media_id'] =  int(df_tmp.iloc[0]['media_id'])
        tmp1_dic['item_id'] = int(df_tmp.iloc[0]['item_id'])
        tmp1_dic['organization_group_id'] = int(df_tmp.iloc[0]['organization_group_id'])
        #x,y座標データ取得
        xy_lst = df_tmp[['x','y']].values.tolist()
        for xy in xy_lst:
            tmp_lst.append(xy)
        tmp1_dic['coordinate'] = tmp_lst
        tmp2_dic[media_id] = tmp1_dic

        #json保存
        savepath = './data/' + media_id + '.json'
        with open(savepath, 'w') as outfile:
            json.dump(tmp1_dic, outfile)
        
        #GCSへアップロード
        bucket_name = 'heatmap-staging'
        source_file_name = './data/' + media_id + '.json'
        destination_blob_name = 'medium_items/media/' + media_id + '/heatmap.json'
        upload_blob(bucket_name, source_file_name, destination_blob_name)
        print('upload完了')
        os.remove(source_file_name) #ローカルで作成したjsonファイル削除
    #元データ削除
    os.remove(destination_file_name)


if __name__ == '__main__':
    main()