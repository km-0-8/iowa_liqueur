# 機械学習モデル作成報告レポート

## 1. 背景と目的
目的は、アイオワ州のリキュール販売データをもとに、売上や傾向を予測・可視化し、解釈性の高い機械学習モデルを構築することです。  
そのために、特徴量の整理や前処理を通じてノイズの少ない学習データを作成し、後段の予測タスクに活かす基盤を整えることを主眼としています。

## 2. データ概要
データは以下のサイトより取得されたアイオワ州の酒類販売履歴です：  
[https://data.iowa.gov/Sales-Distribution/Iowa-Liquor-Sales/m3tr-qhgy/about_data](https://data.iowa.gov/Sales-Distribution/Iowa-Liquor-Sales/m3tr-qhgy/about_data)

主なカラム：
- `date`: 注文日
- `store_number`, `store_name`: 店舗情報
- `category`, `category_name`: 酒類カテゴリ
- `vendor_number`, `vendor_name`: ベンダー情報
- `item_number`, `item_description`: 商品情報
- `bottle_volume_ml`, `state_bottle_cost`, `state_bottle_retail`: 単価やボトル容量
- `bottles_sold`, `sale_dollars`: 売上数量・金額
- `store_location`, `county`, `county_number`: 地理情報

レコード数：約3,000万件

## 3. 処理ステップ
以下のような順で処理が行われています：
1. データ取得（BigQueryより取得）
2. 不要カラムの削除
3. 欠損値処理
4. 冗長なカラムの削除
5. 前処理された中間テーブルの生成

## 4.使用ライブラリ
- データ操作：pandas,dask
- モデル：LinearRegression
- 可視化：matplotlib,seaborn
import matplotlib.pyplot as plt
import seaborn as sns
## 4. 前処理
- **不要なカラム削除**
データ数を減らすため以下のカラムを削除した。
  - `invoice_and_item_number`： 各itemに関する説明分のカラムでありユニークな値のため粒度が高すぎて意味を持たないため
  - `volume_sold_gallons`：他の特徴量から導出される冗長な情報なため（ボトル容量（ml）×販売ボトル数）/3785.411784）
  - `store_location`：このコードから緯度・軽度が算出できるが、今回はaddressなどから位置情報を得る予定のためデータの思いこちらのカラムは削除する

- **欠損値処理**
  - 位置情報がすべて欠損しているカラム(["address", "city", "zip_code", "county_number", "county"])：位置情報の相互補完の余地がない。かつ、欠損数が8万/3000万レコードと全体の0.3%程度であるためそのまま削除する。
  - `county`：同一の`address`を持つレコードから補完。補完できなかった残りの5千/3000万は削除
  - `category` 、`category_name`：両方欠損しているレコードは相互補完の余地がない、かつ、欠損数が2万/3000万レコードと全体の0.05%程度であるためそのまま削除する。
  -その他null列`["vendor_number", "vendor_name", "state_bottle_cost", "state_bottle_retail","sale_dollars"]`：欠損数をすべて足しても全体のレコード数に対して0.08%しか影響がないためそのまま削除する

- **重複確認**：重複レコードなし
- **表記ゆれ処理と冗長なカラムの削除**
  - `store_name` 、`store_number` ：表記ゆれ有 → 同一の`store_number`が同一の`store_name`を指していることをデータ表示し目視で確認→`store_name`が冗長なカラムであると判断し削除
  - `county` 、`county_number` ：表記ゆれ有 → 同一の`county_number`が同一の`county`を指していることをデータ表示し目視で確認→本来であれば`county_number`を残したいが欠損値が700万行と多く補完によりノイズとなる可能性があるため、`county`を残し`county_number`を削除
  - `category` 、`category_name` ：表記ゆれ有 → 同一の`category`が同一の`category_name`を指していることをデータ表示し目視で確認→`category_name`が冗長なカラムであると判断し削除
  - `vendor_number` 、`vendor_name` ：表記ゆれ有 → 同一の`vendor_number`が同一の`vendor_name`を指していることをデータ表示し目視で確認→`vendor_name`が冗長なカラムであると判断し削除
  - `item_number` 、`item_description` ：表記ゆれ有 → 同一の`item_number`が同一の`item_description`を指していることをデータ表示し目視で確認→`item_description`が冗長なカラムであると判断し削除

  - その他の冗長なカラム
    - `["address"、"zip_code"]`：`county`で代替可能、かつ粒度が高すぎて特徴量として意味を持たないため削除
    - `["city"]`：`county`で代替可能なため削除
## 5. 特徴量エンジニアリング
以下のカラムは削除：
- `invoice_and_item_number`: 粒度が細かすぎてモデルにノイズ
- `volume_sold_gallons`: `volume_sold_liters` と重複情報
その他：
- 地理・カテゴリ・価格・数量などを適切に残し、特徴量の精選が行われています。

