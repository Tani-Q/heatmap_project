# nurve-heatmap
実行方法
> python  heatmap_maker.py [カラム名] [item_id/media_id]
カラム名に'item_id'か'media_id'を設定、この後ろにitem_idかmedia_idを設定し実行。
例)
$ python heatmap_maker.py item_id 7031202

実行したら以下のようなメッセージが表示される。
　TDからデータ取り込み開始
　データ取り込み完了
　item_id:7360807/media_id:16251021
　座標補正完了
　ヒートマップデータ作成完了
　凡例付きヒートマップ作成完了
　合成用ヒートマップ作成完了
　余白削除処理完了
　ext: JPG
　Blob medium_items/media/16251021/2048x1024.JPG downloaded to ./imgdata/2048x1024.jpg.
　オリジナル画像読み込み完了
表示が終了したらGCSの/rent-heatmap/medium_items/media/[media_id]/heatmap.jpgが保存されている。
なお、作業に使われたdirはアップロード終了後削除される。
