# -*- coding:utf-8 -*-

#ライブラリ
import sys
import google.cloud.storage
import os
from google.cloud import storage
from google.cloud import storage as gcs
import pandas as pd
from io import BytesIO
import re
import json
import tempfile

#---------------------------- method --------------------------------------
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

"""def download_blob(bucket_name,file_name,project_name):
    # create gcs cliaent
    client = gcs.Client(project_name)
    bucket = client.get_bucket(bucket_name)
    # create blob
    blob = gcs.Blob(file_name, bucket)
    content = blob.download_as_string()
    return pd.read_csv(BytesIO(content))"""

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

 def loop_check(data):
        print("data['name']:{}".format(data['name']))
        if data['name'] == 'logging/heatmap.csv.gz':
          return True  
#---------------------------- method --------------------------------------

def main(data,context):
    print('--- Start!!--- ')
    #無限ループ阻止
    if not loop_check(data):
     sys.exit(1)

    #ディレクトリ準備
    print('Dir作成')
    tmpdir = tempfile.TemporaryDirectory()
    dir_name = tmpdir.name

    #元データダウンロード
    #データダウンロード
    print('ファイルダウンロード開始')
    bucket_name = "heatmap-staging-csv"
    #file_name = "logging/heatmap.csv"
    project_name = "nurve-cloud-98b3f"
    source_blob_name = "logging/heatmap.csv.gz"
    file_name = dir_name + "/heatmap.csv.gz" 

    #blobからjsonファイル取得
    download_blob(bucket_name,source_blob_name,file_name)
    #os.remove(file_name)
    #print('ファイル削除')
    #tmpdir.cleanup()
    #print('Dir削除')
    #df = download_blob(bucket_name,file_name,project_name)
    

    #データ処理
    print('データ処理開始')
    df = pd.read_csv(file_name, sep=',')

    media_id_lst = df['media_id'].values.tolist()
    unique_media_id_lst = list(set(media_id_lst))

    #media_id別データ選別

    ##ディレクトリ準備
    #tmpdir = tempfile.TemporaryDirectory()
    #dir_name = tmpdir.name

    print('media_id別ファイル処理中')
    for media_id in unique_media_id_lst:
        tmp_dic1 = {}
        tmp_dic2 = {}
        tmp_lst = []
    
        #データ選別
        df_tmp = df[df['media_id'] == media_id ].reset_index(drop=True)
        
        media_id = str(int(df_tmp.iloc[0]['media_id']))
        print('media_id:{}'.format(media_id))
        #tmp_dic['media_id'] =  int(df_tmp.iloc[0]['media_id'])
        tmp_dic1['item_id'] = str(int(df_tmp.iloc[0]['item_id']))
        tmp_dic1['organization_group_id'] = str(int(df_tmp.iloc[0]['organization_group_id']))
        #x,y座標データ取得
        xy_lst = df_tmp[['x','y']].values.tolist()
        for xy in xy_lst:
            tmp_lst.append(xy)
        tmp_dic1['coordinate'] = tmp_lst
        tmp_dic2[media_id] = tmp_dic1
    
        #json保存
        savepath = dir_name + '/' + media_id + '.json'
        #savepath = media_id + '.json'
        with open(savepath, 'w') as outfile:
            json.dump(tmp_dic1, outfile)
        
        #GCSへアップロード
        bucket_name = 'heatmap-staging'
        source_file_name = savepath
        destination_blob_name = 'medium_items/media/' + media_id + '/heatmap.json'
        upload_blob(bucket_name, source_file_name, destination_blob_name)
        print('upload完了')
        os.remove(source_file_name) #ローカルで作成したjsonファイル削除
    #ディレクトリ解除
    tmpdir.cleanup()
    print('仮想ディレクトリ削除')

if __name__ == '__main__':
    main()
