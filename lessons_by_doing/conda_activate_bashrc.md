### 콘다 로그인시 환경 activate[https://stackoverflow.com/a/56162704]

.bashrc 는 로그인하거나 하면 실행되는 startup script인 듯 하다
그러나 그냥 명령어 치듯이하면 안되는 것들이 있는데 conda activate \[환경\] 이 그중 하나인 듯 함

```bash
if [ -f "/home/deftson/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/home/deftson/miniconda3/etc/profile.d/conda.sh"
    conda activate torch171
fi
```
