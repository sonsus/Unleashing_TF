http://emflant.tistory.com/127      
http://cpdev.tistory.com/51    
https://stackoverflow.com/questions/6565357/git-push-requires-username-and-password

### 1. `-h` after any command will show lots of things rather than `--help`  
### 2. to ignore folders make `.gitignore` file:    
  - `$vi {workingdirectory}/.gitignore`
  - write:  `/some/path/to/file.txt`, `/a/**/b/`
  - `/a/**/b/` will exclude any directory starts from `/a/` and ends with `b/`
  - `**/file.txt` works similarly    
### 3. `git pull origin seonil222`:     
  - pull `origin/seonil222` to local current branch    
     
     

### 4. revert the repository (wiping out all the changes: DANGER for collabo. work)
  - `git reset --hard <old-commit-id>`
  - `git push -f <remote-name> <branch-name>`

### 5. skip typing username/pw for each git push? nahh...    
``` 
$ git config credential.helper store    
$ git push https://github.com/repo.git   
     
Username for 'https://github.com': <USERNAME>    
Password for 'https://USERNAME@github.com': <PASSWORD>    
```

