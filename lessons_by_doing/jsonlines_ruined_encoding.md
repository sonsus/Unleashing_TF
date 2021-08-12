가끔 
```python
with jsonlines.open("~~", "w") as writer:
  writer.write({~~~~~})
```

했을 때 인코딩이 조져지는걸 볼 수 있다.
터미널에서만 그렇게 보이는게 아니라 실제로 파이썬으로 읽으면 UTF-8로 제대로 읽히던게 깨져서
<code>u"\u00a0"--> </code> 으로 읽혀야 할게 <code>"u\u00a0"-->\xa0</code> 이렇게 읽히는 경우가 있음

language pack이 깔려있지 않은 경우에 간혹 발생하는 문제인 듯 하다.
그래서 아래처럼 설치해주면 해결!

```bash
sudo aptitude install language-pack-en language-pack-gnome-en language-pack-en-base language-pack-gnome-en-base  manpages
```
