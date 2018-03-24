Keras anaconda backend
======================================     
### Answer: Configure activate.d
`~/anaconda3/envs/kwy/etc/conda/activate.d/keras_activate.sh`    

has some scripts inside it    
change it to

```
keras_activate.sh:

export KERAS_BACKEND=tensorflow
```
and reactivate the env

`$source deactivate`   
`$source activate (envname)`  



   
     
     
### configuring `.keras/keras.json`  
- good way to start but not for anaconda
