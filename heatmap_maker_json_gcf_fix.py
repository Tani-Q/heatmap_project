# -*- coding:utf-8 -*-
import sys
import csv
import gzip
import json
import math
import datetime
import gc

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

import cv2
from PIL import Image

import os

#import tdclient
#import pandas_td as td
 
import google.cloud.storage
from google.cloud import storage

import re

import shutil

import tempfile

from flask import escape
import time

#-------------------------------- メソッド -------------------------------------

def json_paser(json):
    json_open = open(json, 'r')
    json_load = json.load(json_open)

    print(json_load)
    

#座標補正メソッド
def corrected_data(line):
    xy_lst = []
    #　小数点以下切り上げ
    #line[1] = math.ceil(float(line[1]))
    #line[2] = math.ceil(float(line[2]))
    x = math.ceil(float(line[2]))
    y = math.ceil(float(line[3]))
    
    #x軸補正
    #if line[1] < 0:
    if x < 0:    
        #xy_lst.append(line[1]*(-1) + 90)
        xy_lst.append(x*(-1) + 90)
    else:
        #xy_lst.append((line[1] - 90)*(-1))
        xy_lst.append((x - 90)*(-1))
     #補正無しX軸   
    #xy_lst.append(line[1])
    
    #y軸補正
    #fix_y = line[2] + 180
    fix_y = y + 180
    if fix_y > 360 :
        xy_lst.append(fix_y - 360)
    else:
        xy_lst.append(fix_y)
    return xy_lst

#json用座標補正メソッド
def corrected_data_json(line):
    xy_lst = []
    #　小数点以下切り上げ
    #line[1] = math.ceil(float(line[1]))
    #line[2] = math.ceil(float(line[2]))
    x = math.ceil(float(line[0]))
    y = math.ceil(float(line[1]))
    
    #x軸補正
    #if line[1] < 0:
    if x < 0:    
        #xy_lst.append(line[1]*(-1) + 90)
        xy_lst.append(x*(-1) + 90)
    else:
        #xy_lst.append((line[1] - 90)*(-1))
        xy_lst.append((x - 90)*(-1))
     #補正無しX軸   
    #xy_lst.append(line[1])
    
    #y軸補正
    #fix_y = line[2] + 180
    fix_y = y + 180
    if fix_y > 360 :
        xy_lst.append(fix_y - 360)
    else:
        xy_lst.append(fix_y)
    return xy_lst


#余白削除メソッド
def margin_cut(file_name='./hm.jpg',output_name='./hm_edge.png'):
    filename = file_name #'./image_3453421.jpg'
    img = cv2.imread(filename)
    img2 = cv2.imread(filename)

    # Grayscale に変換
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2 値化
    ret, thresh = cv2.threshold(img_gs, 250, 255, cv2.THRESH_BINARY)

    # 輪郭を取得
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # i = 1 は画像全体の外枠になるのでカウントに入れない
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(1, len(contours)):
        # ret の中身は (x, y, w, h)
        ret = cv2.boundingRect(contours[i])
        x1.append(ret[0])
        y1.append(ret[1])
        x2.append(ret[0] + ret[2])
        y2.append(ret[1] + ret[3])

    x1_min = min(x1)
    y1_min = min(y1)
    x2_max = max(x2)
    y2_max = max(y2)

    # 枠取りをした結果を表示
    #cv2.rectangle(img, (x1_min, y1_min), (x2_max, y2_max), (0, 255, 0), 2)
    #cv2.imwrite('cropped_edge_rectangle.jpg', img)

    # ギリギリで切り出し
    crop_img = img2[y1_min:y2_max, x1_min:x2_max]
    cv2.imwrite(output_name, crop_img)

    #画像メモリ開放     # 2020/5/13追加
    del crop_img
    gc.collect()

#GCSからの画像データダウンロード
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    """
    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    ) 
    """


#データ量計測
def data_len(*lst):
    lines = list(lst) #list形式に変換一次
    #print(type(lines))
    line_s = lines[0] #list形式に変換二次
    return len(line_s)
        

#ヒートマップデータ作成
def data_maker(*axis_data_lst):
    lst_2d = [] #ヒートマップデータ配列
    x_ranges = [] #x軸bin範囲配列
    x_cnt = [] #x軸bin配列

    #x軸範囲設定
    i = 0
    for x in range(0,360,18):
        x_cnt.append(i)
        x_ranges.append(x)
        i += 1
    
    #x軸データ管理配列設定   
    x_length = len(x_cnt)    

    #x:yデータ集計
    for i in range(0,180,10):
        x_lst = [0] * x_length #x_lstを0初期化
        for xy in  axis_data_lst:
            x = xy[0]
            if x >= i and x < i + 10:
                for cnt, x_range in zip(x_cnt, x_ranges):
                    if xy[1] >= x_range and xy[1] < x_range + 18:
                            x_lst[cnt] += 1
        lst_2d.append(x_lst)
    return lst_2d

#v_max設定
def v_max_set(*lst_2d):
    x_min = 1000
    x_max = 0
    for xs in lst_2d:
        for x in xs:
            if x_max < x:
                x_max = x
            if x_min > x:
                x_min = x

    # vのレンジ幅の設定
    if x_max >= 500:
        v_max = 350
    elif x_max >= 150:
        v_max = 100
    elif x_max >= 100:
        v_max = 100
    elif x_max >= 50:
        v_max = 50
    elif x_max >= 25:
        v_max = 15
    elif x_max >= 5:
        v_max = 5
    else:
        v_max = x_max
    return v_max

#CDSバケット内リスト取得、拡張子抽出
def extension_get(bucket_name,prefix):
    storage_client = storage.Client()

        # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(
            bucket_name, prefix=prefix, delimiter='/'
        )

    blob_lst = []
    target = prefix+'2048x1024.'

    for blob in blobs:
        blob_lst.append(blob.name)
    l_in = [s for s in blob_lst if target in s] 

    if len(l_in) > 0:
        extension = l_in[0].replace(target,'')
        #print('ext: {}'.format(extension))
        return extension
    else:
        return False

#バケット内のオブジェクトリスト取得    
def list_blobs(bucket_name):
        """Lists all the blobs in the bucket."""
        # bucket_name = "your-bucket-name"

        storage_client = storage.Client()

        # Note: Client.list_blobs requires at least package version 1.17.0.
            
        bucket = storage_client.get_bucket('rent-production')
        blobs = bucket.list_blobs(prefix="medium_items/media/", delimiter="/")
        dirs = []
        for page in blobs.pages:
            dirs.extend(page.prefixes)

        [ print(x) for x in dirs ]

#jsonで処理するタイプ    
def heatmap_maker_json(file,dir_name):
    #-----------------------------------  json処理  -------------------------------------------------
    #print('read json.')
    df_s = pd.read_json(file)


    #読み込み画像データ設定
    #print('check media_id.')
    num = re.search(r'[\d]+\.json$',file)
    media_id = num.group().replace('.json','')

    #jsonファイル削除     # 2020/5/13追加
    os.remove(file)

    #item_id =  df_s.loc[0]['item_id']
    #organization_group_id =  df_s.loc[0]['organization_group_id']
    lines =  df_s['coordinate'].values.tolist()
    
    #df削除     # 2020/5/13追加
    del df_s
    gc.collect()
    
    """#読み込み画像データ設定
    line = lines[0]
    file = str(int(line[1])) #item_id
    media_id = str(int(line[0])) #media_id"""

    #print('media_id:{}'.format(media_id))

    #------------------------------ データ作成 -------------------------------------
    #座標補正処理 Ver02
    axis_data_lst = []
    f_lst = []
    idx = 0
    for line in lines:    
        data = corrected_data_json(line) #座標補正
        axis_data_lst.append(data)
    #print('座標補正完了')

    #ヒートマップデータ作成Ver02
    lst_2d = data_maker(*axis_data_lst)
    #print('ヒートマップデータ作成完了')

    #---------------------------------- ヒートマップ作成 ---------------------------------------------
    # vのレンジ幅の調整
    v_max = v_max_set(*lst_2d)
    #print('vmax:{}'.format(v_max))


    #凡例付きヒートマップ作成
    hm_name_sample = dir_name + "/hm_org.jpg"
    plt.figure(figsize=(40,16)) 
    #fig, ax = plt.subplots(figsize=(32,16))
    plt.tick_params(labelsize=30)
    sns.heatmap(lst_2d,vmin=0, vmax=v_max)
    #sns.heatmap(lst_2d,vmin=0, vmax=5)
    plt.savefig(hm_name_sample )
    #print('凡例付きヒートマップ作成完了')

    # 2020/5/13追加
    plt.close()
    gc.collect()

    #合成用ヒートマップ作成
    hm_name = dir_name + "/hm.png" #<- dirをつける
    plt.figure(figsize=(32,16)) 
    sns.heatmap(lst_2d,vmin=0, vmax=v_max,yticklabels=False,xticklabels=False,cbar=False)
    plt.savefig(hm_name)
    #print('合成用ヒートマップ作成完了')
    
    # 2020/5/13追加
    plt.close()
    gc.collect()

    #HM余白削除
    outfile = dir_name + '/hm_edge.png'
    #outfile = margin_cut(hm_name,outfile)
    margin_cut(hm_name,outfile)
    #print('余白削除処理完了')

    
    #--- ここからオミット対象 ---
    #------------------------------------- GCSからの画像データ読み込み -------------------------------------------------
    #直接画像読み込み
    #ファイルの存在確認、拡張子取得
    bucketname = 'rent-production'
    prefix = 'medium_items/media/'+media_id + '/'
    ext = extension_get(bucketname,prefix) #拡張子吸い出し
    if not ext:
        print('拡張子なし')
        return False
    
    #bucket_name = 'rent-production/medium_items/media/'+media_id
    bucket_name = 'rent-production'
    source_blob_name = 'medium_items/media/'+media_id + '/2048x1024.' + ext
    file_name = dir_name + '/2048x1024.jpg'
    try:
        download_blob(bucket_name,source_blob_name,file_name)
    except:
        print('file not found. {}'.format(source_blob_name))
        return False
        
    #print('オリジナル画像読み込み完了')

    #----------------------------------------- 画像処理 ------------------------------------------------------------
    #画像データ読み込み
    filename = dir_name + '/2048x1024.jpg'
    try:
        img = cv2.imread(filename)
        height = img.shape[0] # Errorを引っ掛けるためだけの仕掛
    except  AttributeError:
        print('file not found. {}'.format(filename))
        return False

    #オリジナル画像のリサイズ
    height = img.shape[0]
    width = img.shape[1]
    if height != 1024 or width != 2048:
        img = cv2.resize(img , (int(width*(2048.5/width)), int(height*(1024.5/height))))
    #--- ここまでオミット対象 ---
    

    #余白除去後HM読み込み
    filename = dir_name + '/hm_edge.png' # <- dirをつける
    try: 
        img2 = cv2.imread(filename,-1)
        height = img2.shape[0] # Errorを引っ掛けるためだけの仕掛
    except  AttributeError:
        print('file not found.')

    #サイズ調整
    height = img2.shape[0]
    width = img2.shape[1]

    img2_1 = cv2.resize(img2 , (int(width*(2048/width)), int(height*(1024/height))))
    #print('orgサイズ:{}'.format(img.shape))
    #print('リサイズhm:{}'.format(img2_1.shape))
    cv2.imwrite(dir_name + '/resize_heatmap.png', img2_1)
    #print('ヒートマップのサイズ調整完了')

    
    #--- ここから機能オミット ---
    #画像合成
    blended = cv2.addWeighted(src1=img,alpha=0.6,src2=img2_1,beta=0.4,gamma=0.3)

    #合成後画像保存
    cv2.imwrite(dir_name + '/blended_test.jpg', blended) # <- dirをつける
    #print('合成画像作成完了')

    # 2020/5/13追加
    del img2
    del img2_1
    del blended
    gc.collect()
    #--- ここまで機能オミット ---
    
    """
    #半透明画像作成
    im_rgb = Image.open(dir_name + '/resize_heatmap.png')
    im_rgb.putalpha(128)
    im_rgb.save(dir_name + '/pillow_putalpha_solid.png')
    """

    #凡例切り出し準備
    im = Image.open(hm_name_sample)

    #凡例切り出し&resize
    #im_crop = im.crop((2230,120,2350,1030))
    im_crop = im.crop((3150,170,3250,1450)) #2020/04/16 凡例切り出しのズレが生じていたので修正
    im_crop_rsize = im_crop.resize((100,1024))   
    #plt.imshow(im_crop_rsize)
    im_crop_rsize.save(dir_name + '/colorbar_crop.jpg', quality=100) # <- dirをつける
    #print('凡例作成完了')

    # 2020/5/13追加
    del im
    del im_crop
    del im_crop_rsize
    gc.collect()

    
    #--- ここから機能オミット ---
    #ブランク画像作成
    height = 1024
    width = 2200
    blank = np.zeros((height, width, 3))
    blank += 255 #←全ゼロデータに255を足してホワイトにする
 
    cv2.imwrite(dir_name + '/blank.jpg',blank) # <- dirをつける
    #print('ベース画像作成完了')


    #凡例付きヒートマップ作成
    img0 = cv2.imread(dir_name + '/blank.jpg') # ブランク画像
    img1 = cv2.imread(dir_name + '/blended_test.jpg') # ヒートマップ合成画像
    img2 = cv2.imread(dir_name + '/colorbar_crop.jpg') # 凡例

    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    base_img = img0


    #ヒートマップ画像貼り付け基準点設定
    x_offset=0
    y_offset=0

    #ヒートマップ画像合成
    base_img[y_offset:y_offset+img1.shape[0], x_offset:x_offset+img1.shape[1]] = img1

    #凡例画像貼り付け基準点設定
    x_offset=2050
    y_offset=0

    #凡例画像合成
    base_img[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1]] = img2
    base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

    #media_dir = './medium_items/media/' + media_id 
    media_dir = dir_name + '/' + media_id + '.jpg' # 凡例付き合成画像

    #if not os.path.exists(media_dir):
        #os.makedirs(media_dir) #dirが無い場合作成

    #dir_file_name = './medium_items/media/' + media_id + '/heatmap.jpg'

    #cv2.imwrite(dir_file_name,base_img)
    cv2.imwrite(media_dir,base_img)
    #print('凡例付き合成画像作成完了')
    #--- ここまで機能オミット ---
    
    #--- ここから機能オミット ---
    #合成ヒートマップ画像アップローダー
    #- 本番環境設定 -
    #bucket_name = 'rent-production'
    #destination_blob_name = 'reports/heat_maps/hm/media/' + media_id + '/heatmap.jpg' 
 

    bucket_name = "heatmap-staging" #<- 将来的には本番環境へ変更
    source_file_name = media_dir # 凡例付き合成画像
    destination_blob_name = 'medium_items/hm/media/' + media_id + '/heatmap.jpg'
    upload_blob(bucket_name, source_file_name, destination_blob_name) #画像データアップロード
    #--- ここまで機能オミット ---

    """
    #半透明ヒートマップ画像アップローダー
    #bucket_name = "rent-production" #"heatmap-staging" #<- 将来的には本番環境へ変更
    bucket_name = "heatmap-staging"
    source_file_name = dir_name + '/pillow_putalpha_solid.png' # 半透明画像
    destination_blob_name = 'medium_items/hm/media/' + media_id + '/pillow_putalpha_solid.png'
    upload_blob(bucket_name, source_file_name, destination_blob_name) #画像データアップロード

    #凡例画像アップローダー
    #bucket_name = "rent-production" #"heatmap-staging" #<- 将来的には本番環境へ変更
    bucket_name = "heatmap-staging"
    source_file_name = dir_name + '/colorbar_crop.jpg' # 凡例画像
    destination_blob_name = 'medium_items/hm/media/' + media_id + '/colorbar_crop.jpg'
    upload_blob(bucket_name, source_file_name, destination_blob_name) #画像データアップロード
    """
    #作成画像削除
    os.remove(dir_name + "/hm_org.jpg")
    os.remove(dir_name + "/hm.png")
    os.remove(dir_name + '/hm_edge.png')
    os.remove(dir_name + '/2048x1024.jpg') # オミット対象
    os.remove(dir_name + '/blended_test.jpg') # オミット対象 
    os.remove(dir_name + '/resize_heatmap.png')
    #os.remove(dir_name + '/pillow_putalpha_solid.png')
    os.remove(dir_name + '/colorbar_crop.jpg') 
    #os.remove(dir_name + '/blank.jpg') # オミット対象
    #os.remove(media_dir) # オミット対象
    

    #return True    
    
#GCSへのデータアップローダー    
def upload_blob(bucket_name, source_file_name, destination_blob_name):
        #Uploads a file to the bucket.
        # bucket_name = "your-bucket-name"
        # source_file_name = "local/path/to/file"
        # destination_blob_name = "storage-object-name"

        storage_client = storage.Client()
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

def down_blob(bucket_name,source_blob_name,file_name):
        #------------------------------------- GCSからの画像データ読み込み -------------------------------------------------
    #直接画像読み込み
    #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/tani_kyuichiro/nurve-cloud-98b3f-bb7c97f8fb03.json'
    
    
    #ファイルの存在確認、拡張子取得
    bucketname = 'rent-production'
    prefix = 'medium_items/media/'+media_id + '/'
    ext = extension_get(bucketname,prefix) #拡張子吸い出し
    if not ext:
        return False
    
    #bucket_name = 'rent-production'
    #source_blob_name = 'medium_items/media/'+media_id + '/2048x1024.' + ext
    #file_name = './tmp/2048x1024.jpg'
    try:
        download_blob(bucket_name,source_blob_name,file_name)
    except:
        print('file not found. {}'.format(source_blob_name))
        return False
        
    #print('オリジナル画像読み込み完了')
    
#バケット内のオブジェクトリスト取得
def list_blobs(bucket_name):
        """Lists all the blobs in the bucket."""
        # bucket_name = "your-bucket-name"

        storage_client = storage.Client()

        # Note: Client.list_blobs requires at least package version 1.17.0.
        blobs = storage_client.list_blobs(bucket_name)
        
        blobs_lst = []
        for blob in blobs:
            #print(blob.name)
            blobs_lst.append(blob.name)
        return blobs_lst    

def loop_check(data):
        print("data['name']:{}".format(data['name']))
        if re.match(r'medium_items/media/[0-9]+/heatmap\.json',data['name']):
          return True  

#------------------------------------ メソッド -----------------------------------------

#def main(data):
def main(request):
    """#強制終了
    if not loop_check(data):
        sys.exit(1)"""
    #media_id受取
    #print('受信開始')
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'data' in request_json:
        media_ids = request_json['data']
        cnt = request_json['cnt']
        lcnt = request_json['lcnt']
    elif request_args and 'data' in request_args:
        media_ids = request_args['data']
        cnt = request_json['cnt']
        lcnt = request_json['lcnt']
    else:
        media_ids = False
        return escape(media_ids)  

    #ディレクトリ準備
    #print('Dir作成')
    tmpdir = tempfile.TemporaryDirectory()
    dir_name = tmpdir.name
    """
    #media_idのファイル名リスト作成
    blobs_lst = list_blobs('heatmap-staging')
    json_lst = []
    for blob in blobs_lst:
        if re.search(r'medium_items/media/[0-9]+/heatmap\.json',blob):
            json_lst.append(blob)
    """

    #blobからjsonファイル取得してヒートマップ作成
    #for json in json_lst:
    #json = data['name']
    #num = re.search(r'\d+',json)
    #media_id = num.group()
    #bucket_name = 'rent-production' #'heatmap-staging'
    bucket_name = 'heatmap-staging'
    #source_blob_name = 'reports/heat_maps/media/' + media_id + '/coordinate.json' #'medium_items/media/' +media_id + '/heatmap.json'
    
    #------- debug -------
    test_lst = [14687738, 16287716, 13809236, 13808540, 13809014, 16376628, 14688089] 
    #------- debug -------  
    llcnt = 1
    for media_id in media_ids:
        if media_id == '':
            continue
        #debug
        if media_id in test_lst:
            print('----- Hit!!---- {}'.format(media_id))
        #debug
        
        source_blob_name = 'medium_items/media/' +str(media_id) + '/heatmap.json'

        #source_blob_name = json
        file_name = dir_name + '/' + str(media_id) + '.json'
        #blobからjsonファイル取得
        #print('json取得開始')
        download_blob(bucket_name,source_blob_name,file_name)
    
        #ヒートマップ作成
        #print('ヒートマップ作成開始')
        #print('m_id:{}'.format(media_id))
        heatmap_maker_json(file_name,dir_name)
    
        #os.remove(file_name) #json削除 #2020/5/13 機能停止
        
        #print('cnt:{}/lcnt:{}/llcnt:{}'.format(cnt,lcnt,llcnt))
        
        if len(media_ids) <= llcnt:
            print('cnt:{}/lcnt:{}/llcnt:{}/mcnt:{}'.format(cnt,lcnt,llcnt,len(media_ids)))
            if len(media_ids) < 60:
                print('--cnt:{}/lcnt:{}/llcnt:{}/mcnt:{}'.format(cnt,lcnt,llcnt,len(media_ids)))
            return escape(media_id)
        
        time.sleep(1)
        llcnt += 1

    tmpdir.cleanup() #仮想dir削除

    #return escape(media_ids)

