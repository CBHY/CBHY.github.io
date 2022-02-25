# Git 09



## Push & Pull

### Push(Remote -> Local)

```bash
git push
```

#### 	git push --set-upstream (원격) (로컬) 명령어로 이미 두 branch가 연결되어 있기 때문에 Only, git push



### Pull(Local -> Remote)

```bash
git pull
```

#### 	각 branch 상관 없이 모든 변동사항을 commit 단위로 remote에서 local로 받아옴.



## 충돌 해결

### 1. Pull 할 내용이 있을 때, Push 하는 경우

#### 	Git에서는 무조건 Pull을 이용해서 최신화 된 상황 이후에 Push가 가능,

```bash
# merge 방식 Pull
git pull --no-rebase
git push
```

#### 	local의 main과 remote의 main을 각각 다른 branch로 보고 두 branch에서의 commit들을 merge

```bash
# rebase 방식 Pull
git pull --rebase
git push
```

#### 	remote의 branch를 먼저 끌어와 실행한 후 local 의 변화를 실행



### 2.  협업 상 충돌 발생

### 	reset, revert, rebase, merge 상황 시 충돌과 비슷하게 충돌 해결 이후 똑같이 진행



### 3. GitHub의 내용에 문제가 생겼을 경우

```bash
# 로컬 내역을 강제로 원격에 Push
git push --force
```



## 원격 Branch 관리

### git fetch

```bash
git fetch
```

#### 	원격의 branch를 local에서도 확인하고 관리하기 위해서 git fetch 명령어로 최신화 해주어야 함.

```bash
git switch -t origin/(branch)
```

#### 	원격 branch와 이름이 같은 branch를 local에 생성하고 연결.(git push --set-upsetram과 반대방향)

```bash
# 원격 branch 삭제
git push origin --delete (원격 branch)
```

