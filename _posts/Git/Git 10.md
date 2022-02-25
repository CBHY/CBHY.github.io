# Git 10



## Commit 단위 이동

### HEAD : 현재 속한 Branch의 가장 최신 Commit

```bash
# HEAD를 기준 Commit 단위 이동
git checkout HEAD^
# 또는
git checkout HEAD~
```

#### 	^ or ~ 하나 당 HEAD 뒤로 한 commit씩 이동

```bash
# hash를 이용한 Commit 단위 이동
git checkout (hash)
```

