#ライブラリ
import os
import sys
from google.cloud import storage as gcs
import pandas as pd
from io import BytesIO

#from google.cloud import storage

import re

import json
import tempfile
import gc

import asyncio
import requests
from requests.exceptions import Timeout

from flask import escape
import time
import pprint

#GCSへのデータアップローダー    
def upload_blob(bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        # bucket_name = "your-bucket-name"
        # source_file_name = "local/path/to/file"
        # destination_blob_name = "storage-object-name"

        #storage_client = storage.Client()
        storage_client = gcs.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)
        """
        print(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )
        """
def download_blob(bucket_name,file_name,project_name):
    # create gcs cliaent
    client = gcs.Client(project_name)
    bucket = client.get_bucket(bucket_name)
    # create blob
    blob = gcs.Blob(file_name, bucket)
    content = blob.download_as_string()
    return pd.read_csv(BytesIO(content))
"""
def download_blob(bucket_name, source_blob_name, destination_file_name):
    #Downloads a blob from the bucket.
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
"""

#----------------------- 非同期 ------------------------

async def req_url(cnt,lcnt,*media_id):
    print('非同期開始')
    loop = asyncio.get_event_loop()
    
    try: 
        response = requests.put(
                'https://asia-northeast1-nurve-cloud-98b3f.cloudfunctions.net/heatmap_maker',
                json.dumps({'data' : media_id, "cnt" : cnt, "lcnt" : lcnt}),
                headers={'Content-Type': 'application/json'}, timeout=2)
        print(response)
        #loop.run_in_executor(None, response)        
    except Timeout:
            #print('{} continued'.format(media_id))
            print('continued')

async def main_loop(cnt,lcnt,*media_id):
    #loop = asyncio.get_event_loop()
    
    #loop.run_until_complete(asyncio.gather(*[req_url(x) for x in media_id_lst]))
    await req_url(cnt,lcnt,*media_id)


def nom_req_url(*media_id):
    #print('同期開始')
    
    try: 
        response = requests.put(
                'https://asia-northeast1-nurve-cloud-98b3f.cloudfunctions.net/heatmap_maker',
                json.dumps({'data' : media_id}),
                headers={'Content-Type': 'application/json'})#, timeout=0.8)
        print(response)        
    except Timeout:
            print('{} continued'.format(media_id))

#------------------------- 非同期 -------------------------

def main(request):
    #media_id受取
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'data' in request_json:
        media_ids = request_json['data']
        cnt = request_json['cnt']
    elif request_args and 'data' in request_args:
        media_ids = request_args['data']
        cnt = request_json['cnt']
    else:
        media_ids = False
    
    """
    #データダウンロード
    print('ファイルダウンロード開始')
    bucket_name = "heatmap-staging-csv"
    #file_name = "logging/heatmap.csv"
    project_name = "nurve-cloud-98b3f"
    source_blob_name = "logging/heatmap.csv"
    file_name = dir_name + "/heatmap.csv" 
        #blobからjsonファイル取得
    download_blob(bucket_name,source_blob_name,file_name)

    os.remove(file_name)
    
    #ディレクトリ解除
    tmpdir.cleanup()
    print('仮想ディレクトリ削除')
    """

    #データダウンロード
    #print('ファイルダウンロード開始')
    bucket_name = "heatmap-staging-csv"
    file_name = "logging/heatmap.csv"
    project_name = "nurve-cloud-98b3f"
    df = download_blob(bucket_name,file_name,project_name)



    """
    bucket_name = "heatmap-staging-csv"
    file_name = "logging/heatmap.csv"
    project_name = "nurve-cloud-98b3f"
    df = download_blob(bucket_name,file_name,project_name)
    """
    #データ処理
    loop = asyncio.get_event_loop()
    #print('データ処理開始')
    #debug
    #cnt = 1
    for m in media_ids:

        tmp_dic1 = {}
        #tmp_dic2 = {}
        tmp_lst = []
    
        #データ選別
        df_tmp = df[df['media_id'] == m ].reset_index(drop=True)
    
        #del df #dfの削除
        #gc.collect()    
    
        media_id = str(int(df_tmp.iloc[0]['media_id']))
        #print('media_id:{}'.format(media_id))
        #tmp_dic['media_id'] =  int(df_tmp.iloc[0]['media_id'])
        #tmp_dic1['item_id'] = int(df_tmp.iloc[0]['item_id'])
        #tmp_dic1['organization_group_id'] = int(df_tmp.iloc[0]['organization_group_id'])
        #x,y座標データ取得
        xy_lst = df_tmp[['x','y']].values.tolist()
        del df_tmp #dfの削除
        gc.collect() 
        for xy in xy_lst:
            tmp_lst.append(xy)
        tmp_dic1['coordinate'] = tmp_lst
        #tmp_dic2[media_id] = tmp_dic1

        #ディレクトリ準備
        #print('Dir作成')
        tmpdir = tempfile.TemporaryDirectory()
        dir_name = tmpdir.name

        #json保存
        savepath = dir_name + '/' + media_id + '.json'
        #savepath = media_id + '.json'
        with open(savepath, 'w') as outfile:
            json.dump(tmp_dic1, outfile)
    
        #GCSへアップロード
        #bucket_name = 'rent-production'
        bucket_name = 'heatmap-staging'
        source_file_name = savepath
        #destination_blob_name = 'reports/heat_maps/media/' + media_id + '/coordinate.json' #'medium_items/media/' +media_id + '/heatmap.json'
        destination_blob_name = 'medium_items/media/' +media_id + '/heatmap.json'
        upload_blob(bucket_name, source_file_name, destination_blob_name)
        #print('upload完了')
        os.remove(source_file_name) #ローカルで作成したjsonファイル削除
        #ディレクトリ解除
        tmpdir.cleanup()
        #print('仮想ディレクトリ削除')
        m = str(m)
    #print("media_id_lst : {}".format(media_ids))

    #n個のmedia_idをlistにして固める
    CHNK =  59 #499
    chunks = []
    lst_cnt = len(media_ids)
    #print(lst_cnt)

    for i in range(0,lst_cnt,CHNK):
        #if len(unique_media_id_lst) >= CHNK + 1:
        ch = media_ids[0:CHNK+1]
        ch1 = [a for a in ch if a != '']
        if len(ch1) >= 1:
            #ch1 = [a for a in ch if a != '']
            chunks.append(ch1)
            del media_ids[0:CHNK+1]
            #print('chunks:{}'.format(chunks))

        #loop.run_until_complete(asyncio.gather(req_url(m)))
        #print('cnt:{}'.format(cnt))
    local_cnt = 1
    test_lst = [14687738, 16287716, 13809236, 13808540, 13809014, 16376628, 14688089]
    for m in chunks:
        i = 1 #degub
        for n in m: #debug
            if n in test_lst: #debug
                print('ヒートマップ作成 cnt:{}/lcnt:{}/len_m:{}/lcnt:{}'.format(cnt,local_cnt,len(m),i))
        loop.run_until_complete(asyncio.gather(req_url(cnt,local_cnt,*m)))
        #loop.run_until_complete(asyncio.gather(main_loop(cnt,local_cnt,*m)))
        #nom_req_url(*m)
        local_cnt += 1
        #i += 1 #debug
        #if local_cnt >= len(chunks):
            #sys.exit(1)
        time.sleep(2)
    #return escape(int(media_id))
