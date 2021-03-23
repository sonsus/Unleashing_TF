jupyter lab --generate-config
jupyter lab password

vi ~/.jupyter/jupyter_lab_config.py

change:
  c.*.password = "hashed password from ~/.jupyter/jupyter_server_config.json here"
  c.ServerApp.ip = "server IP"
  c.*.browser = False
  
  
  
