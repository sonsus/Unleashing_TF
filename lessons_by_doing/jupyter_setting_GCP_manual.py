jupyter_server_setting_GCP.txt

tested on ubuntu1604
SSL secured server is needed

1. configure jupyter

$ conda install jupyter
$ jupyter notebook --generate-config
$ jupyter notebook password
    #hashed passwd is stored in ~/.jupyter/jupyter_notebook_config.json

$ vi ~'yourname'/.jupyter/jupyter_notebook_config.py
    c.NotebookApp.password = '   your hashed password here!    '

$ openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem
    #mycert.pem : cert file
    #mykey.key : key file
    #remember where those were stored

$ vi jupyter_notebook_config.py
    
    #absolute path for cert files
    c.NotebookApp.certfile = '/absolute/path/to/your/certificate/mycert.pem'
    c.NotebookApp.keyfile = '/absolute/path/to/your/certificate/mykey.key'
    
    # write internal ip for the instance here!
    c.NotebookApp.ip = ' internal.ip.address.here  '
    c.NotebookApp.password = u'sha1:bcd259ccf...<your hashed password here>'
    c.NotebookApp.open_browser = False

    # It is a good idea to set a known, fixed port for server access
    c.NotebookApp.port = 9999

2. configure firewall settings 

(1) console --> VPC network --> firewall rules 
(2) create firewall rule
    priority : 1000 (default)
    direction: ingress (default)
    targets  : All instances in the network     ##### important
    source filter: ip ranges (default)
    source IP ranges: 0.0.0.0/0                 ##### important
    protocols and ports:
        tcp:9999                                ##### we configured jupyter to connect thru port 9999
(3) save and wait for a while


3. launch!

(1) $ jupyter notebook
(2) open browser and enter as follows
    https://instance.external.ip.address:9999/
(3) type the password
(4) enjoy!
(5) to run the jupyter notebook after ssh termination, refer to 'remote_running.txt' in this folder

4. if kernels dont show:
    $ conda install nb_conda_kernels


----------------------------------------------------------------------------------------
Refered to... (the last link was the most helpful)

http://jupyter-notebook.readthedocs.io/en/stable/public_server.html#running-a-public-notebook-server
https://www.slideshare.net/HyunsikYoo/ipython-serverjupyter-server
https://jeffdelaney.me/blog/running-jupyter-notebook-google-cloud-platform/
https://medium.com/@parkjonghyeob/google-cloud-%EC%97%90-deep-learning-%EC%A4%80%EB%B9%84%ED%95%98%EA%B8%B0-3c6457cb6f35

