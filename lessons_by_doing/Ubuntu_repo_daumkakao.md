### changing ubuntu repo to daumkakao ftp    
(https://openwiki.kr/tech/ubuntu_daumkakao_repository)   
   
     
```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo sed -i 's/kr.archive.ubuntu.com/ftp.daumkakao.com/g' /etc/apt/sources.list
sudo apt-get update
sudo apt-get upgrade```
