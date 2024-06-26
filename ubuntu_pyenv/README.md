# ubuntu_python

#### 1. 下記コマンドでビルドとコンテナ起動を行う
```
$ root@aa1519d1a400:/workspace# apt install docker
$ root@aa1519d1a400:/workspace# sudo chmod 666 /var/run/docker.sock
$ docker-compose up -d app
```

#### 2. コンテナにアクセスし、ライブラリをインストール
```
$ docker-compose exec  app /bin/bash

root@aa1519d1a400:/workspace# pip install -r requirements.txt
```

#### 3. jupyter を起動
```
root@aa1519d1a400:/workspace#poetry run jupyter lab --allow-root --no-browser --NotebookApp.token='' --port 8888 --ip=0.0.0.0
```
#### 4.http://localhost:8081/ にアクセス

## 番外編　dockerコンテナにvsdcodeをアッタチしてvscodeで開発

#### 1.拡張機能からremote containerをインストール

#### 2.コンテナを起動した後、左のタブにあるRemote expolrerをクリックし、上のremoteをcontainersに変更。

#### 3.CONTAINERS/Dev Containersから起動しているコンテナをドラックして一番左にあるattch to Containerをクリック

#### 4.vscodeが起動したらまずは拡張機能からpythonをインストールして、コマンドパレットを開き、select python interpreterを検索しpoetry直下のpythonを選ぶ
