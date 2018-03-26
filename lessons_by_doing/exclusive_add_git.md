
How to exclude some folder from staging
-----------------------------------------------------  

#### Exclude a folder     
```
$ git add .
$ git reset -- somefolder_to_exclude
```   

#### Exclude a file   
```
$ git add -u
$ git reset -- main/dontcheckmein.txt
```
