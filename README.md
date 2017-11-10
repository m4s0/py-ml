# py-ml

python machine learning

#
use python 3

###### Setup

```
virtualenv -p python3 env
```


###### Activate

```
source env/bin/activate
```


###### Install

```
pip install -r requirements.txt
```


###### Download data

```
wget --output-document ./_not_approved/comments.json http://talk.calciomercato.pro/api/comments-filtered?from=01-10-2017&to=10-11-2017&status=2
```

```
wget --output-document ./_approved/comments.json http://talk.calciomercato.pro/api/comments-filtered?from=05-11-2017&to=10-11-2017&status=1
```

###### Run Code

```
python <FILE_NAME>
```


###### Obtaining the IMDb movie review dataset


The IMDB movie review set can be downloaded from http://ai.stanford.edu/~amaas/data/sentiment/

Move downloaded file to `data` folder.

Decompress the file and convert to csv running `imdb_to_csv.py`
